"""
=============================================================================
Predictive Pipeline: SOH & RUL Estimation for Li-ion Cells
Dataset: NASA PCoE Li-ion Battery Dataset
Model: RandomForestRegressor (scikit-learn)
Author: Machine Learning Engineering / BMS Research
=============================================================================

Script Structure:
    1. Imports and Configuration
    2. Dataset Loading and Parsing
    3. Feature Engineering (Discharge Cycles)
    4. SOH Calculation and RUL Labeling
    5. Temporal / Battery-Split Validation (Preventing Data Leakage)
    6. scikit-learn Pipelines (Scaler + RandomForest)
    7. Future work  - Hyperparameter Optimization (RandomizedSearchCV)
    8. Evaluation Metrics: RMSE, MAE, R²
    9. Visualizations: Predicted vs. Actual across Cycles
    10. Feature Importance Analysis
"""

# =============================================================================
# 1. IMPORTS AND CONFIGURATION
# =============================================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio

from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress warnings for cleaner terminal output during training
warnings.filterwarnings("ignore")
np.random.seed(42)

# --- BMS Domain Global Parameters ---
SOH_EOL_THRESHOLD = 0.80               # End-of-Life: 80% of nominal capacity
NOMINAL_CAPACITY_AH = 2.0              # Typical nominal capacity for the NASA dataset (Ah)
CUTOFF_VOLTAGE = 2.7                   # Discharge cutoff voltage (V)
DATA_FOLDER_NAME = Path("./data_mat")  # Directory containing NASA PCoE .mat files

# Train-Test Split Definitions
# Training on multiple cells to capture degradation variance, testing on a hold-out cell
BATTERY_IDS_TRAIN = ["B0005", "B0006", "B0018"]
BATTERY_IDS_TEST  = ["B0007"]


# =============================================================================
# 2. DATA LOADING AND NASA PCoE DATASET PARSING (.mat)
# =============================================================================

def load_battery_mat(file_path: str) -> Dict:
    """
    Loads a NASA PCoE .mat file and extracts structured cycle data.

    The NASA PCoE dataset stores each battery's historical data in a .mat file 
    containing a nested struct with 'charge', 'discharge', and 'impedance' profiles.

    Args:
        file_path (str): Path to the battery's .mat file.

    Returns:
        Dict: A dictionary containing separated lists for 'charge' and 'discharge' cycles.
    """
    mat = sio.loadmat(file_path, simplify_cells=True)
    
    root_key = [k for k in mat.keys() if not k.startswith("_")][0]
    battery_struct = mat[root_key]

    cycles_raw = battery_struct["cycle"]
    
    if not isinstance(cycles_raw, (list, np.ndarray)):
        cycles_raw = [cycles_raw]

    discharge_cycles, charge_cycles = [], []

    for cycle in cycles_raw:
        cycle_type = str(cycle.get("type", "")).strip().lower()
        data       = cycle.get("data", {})

        if cycle_type == "discharge":
            discharge_cycles.append({
                "voltage":     np.atleast_1d(data.get("Voltage_measured", [])),
                "current":     np.atleast_1d(data.get("Current_measured", [])),
                "temperature": np.atleast_1d(data.get("Temperature_measured", [])),
                "time":        np.atleast_1d(data.get("Time", [])),
                "capacity":    float(data.get("Capacity", np.nan)),
            })
        elif cycle_type == "charge":
            charge_cycles.append({
                "voltage":  np.atleast_1d(data.get("Voltage_measured", [])),
                "current":  np.atleast_1d(data.get("Current_measured", [])),
                "time":     np.atleast_1d(data.get("Time", [])),
            })

    return {"discharge": discharge_cycles, "charge": charge_cycles}


# =============================================================================
# 3. FEATURE ENGINEERING (DISCHARGE CYCLE EXTRACTION)
# =============================================================================

def extract_discharge_features(cycle: Dict, nominal_capacity: float = NOMINAL_CAPACITY_AH) -> Dict:
    """
    Extracts physical and statistical features from an individual discharge cycle.

    Extracted Features:
        - capacity_ah          : Measured capacity during this cycle (Ah)
        - voltage_drop         : Total voltage drop during the discharge phase
        - voltage_mean         : Mean discharge voltage
        - voltage_std          : Voltage standard deviation (acts as a degradation proxy)
        - temp_max             : Maximum temperature reached (°C)
        - temp_mean            : Mean temperature during discharge
        - temp_delta           : Temperature variation (max - min)
        - discharge_duration_s : Total duration of the discharge phase in seconds
        - time_to_cutoff_s     : Time taken to reach the cutoff voltage (CUTOFF_VOLTAGE)
        - energy_wh            : Estimated delivered energy (integral of V*I*dt)
        - coulombic_efficiency : Ratio of measured capacity to nominal capacity
        - internal_resistance  : Simplified internal resistance estimation (RI = ΔV / ΔI)

    Args:
        cycle (Dict): Dictionary containing NumPy arrays for a single discharge cycle.
        nominal_capacity (float): The battery's nominal capacity in Ah.

    Returns:
        Dict: A dictionary containing the computed numerical features for the cycle.
    """
    V = cycle["voltage"]
    I = cycle["current"]
    T = cycle["temperature"]
    t = cycle["time"]
    cap = cycle["capacity"]

    features = {}

    # --- Capacity and Efficiency ---
    features["capacity_ah"]          = cap if not np.isnan(cap) else np.nan
    features["coulombic_efficiency"] = cap / nominal_capacity if not np.isnan(cap) else np.nan

    # --- Voltage Features ---
    if len(V) > 1:
        features["voltage_drop"] = float(V[0] - V[-1])
        features["voltage_mean"] = float(np.mean(V))
        features["voltage_std"]  = float(np.std(V))
    else:
        features.update({"voltage_drop": np.nan, "voltage_mean": np.nan, "voltage_std": np.nan})

    # --- Temperature Features ---
    if len(T) > 1:
        features["temp_max"]   = float(np.max(T))
        features["temp_mean"]  = float(np.mean(T))
        features["temp_delta"] = float(np.max(T) - np.min(T))
    else:
        features.update({"temp_max": np.nan, "temp_mean": np.nan, "temp_delta": np.nan})

    # --- Temporal Features ---
    if len(t) > 1:
        features["discharge_duration_s"] = float(t[-1] - t[0])

        # Time to reach the threshold cutoff voltage
        cutoff_indices = np.where(V <= CUTOFF_VOLTAGE)[0]
        features["time_to_cutoff_s"] = float(t[cutoff_indices[0]] - t[0]) \
            if len(cutoff_indices) > 0 else float(t[-1] - t[0])
    else:
        features.update({"discharge_duration_s": np.nan, "time_to_cutoff_s": np.nan})

    # --- Delivered Energy (Trapezoidal integration of P = V * |I| dt) ---
    if len(V) > 1 and len(t) > 1 and len(I) > 1:
        min_len = min(len(V), len(I), len(t))
        power   = np.abs(V[:min_len]) * np.abs(I[:min_len])
        features["energy_wh"] = float(np.trapezoid(power[:min_len], t[:min_len]) / 3600)
    else:
        features["energy_wh"] = np.nan

    # --- Estimated Internal Resistance (Ohmic drop over initial samples) ---
    # Utilizes the first 5 samples to estimate the immediate voltage drop
    if len(V) >= 5 and len(I) >= 5:
        delta_v = abs(V[0] - V[4])
        delta_i = abs(I[0] - I[4])
        features["internal_resistance"] = float(delta_v / delta_i) if delta_i > 1e-6 else np.nan
    else:
        features["internal_resistance"] = np.nan

    return features


def build_feature_dataframe(battery_data: Dict, battery_id: str) -> pd.DataFrame:
    """
    Constructs a feature DataFrame for all discharge cycles of a given battery,
    laying the groundwork for SOH calculation and RUL labeling.

    Args:
        battery_data (Dict): The parsed dictionary returned by load_battery_mat().
        battery_id (str): The unique identifier for the battery (e.g., 'B0005').

    Returns:
        pd.DataFrame: A DataFrame with one row per valid discharge cycle.
    """
    records = []

    for cycle_idx, cycle in enumerate(battery_data["discharge"]):
        feats = extract_discharge_features(cycle)
        feats["cycle_number"] = cycle_idx + 1
        feats["battery_id"]   = battery_id
        records.append(feats)

    df = pd.DataFrame(records)
    df.dropna(subset=["capacity_ah"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# =============================================================================
# 4. SOH CALCULATION AND RUL LABELING
# =============================================================================

def compute_soh_rul(df: pd.DataFrame,
                    nominal_capacity: float = NOMINAL_CAPACITY_AH,
                    eol_threshold: float    = SOH_EOL_THRESHOLD) -> pd.DataFrame:
    """
    Appends the target variables ('soh' and 'rul') to the feature DataFrame.

    Definitions:
        SOH (State of Health) = Current_Capacity / Nominal_Capacity
        RUL (Remaining Useful Life) = Number of remaining cycles until SOH < eol_threshold

    Methodological Note:
        RUL calculation here is performed offline using a look-ahead approach, 
        leveraging the full known degradation trajectory to generate ground-truth labels. 
        In an online, real-world BMS deployment, an active prognostic model is 
        required to estimate this future EOL cycle.

    Args:
        df (pd.DataFrame)       : DataFrame containing the 'capacity_ah' column.
        nominal_capacity (float): The battery's nominal capacity (Ah).
        eol_threshold (float)   : SOH threshold defining End-of-Life (e.g., 0.80 for 80%).

    Returns:
        pd.DataFrame: The enriched DataFrame containing 'soh', 'rul', and 'is_eol' flags.
    """
    df = df.copy()
    
    # Calculate State of Health (SOH)
    df["soh"] = df["capacity_ah"] / nominal_capacity

    # Identify the End-of-Life (EOL) cycle: the first cycle where SOH drops below the threshold
    eol_mask  = df["soh"] < eol_threshold
    eol_cycle = df.loc[eol_mask, "cycle_number"].min() if eol_mask.any() else df["cycle_number"].max()

    # Calculate Remaining Useful Life (RUL)
    # RUL represents cycles remaining until EOL. Negative values (post-EOL) are clipped to zero.
    df["rul"] = (eol_cycle - df["cycle_number"]).clip(lower=0)
    
    # Flag to easily filter out or analyze post-EOL behavior
    df["is_eol"] = df["soh"] < eol_threshold

    return df


# =============================================================================
# 5. TEMPORAL / BATTERY-SPLIT VALIDATION (DATA LEAKAGE PREVENTION)
# =============================================================================

def split_by_battery(all_df: pd.DataFrame,
                     train_ids: List[str],
                     test_ids:  List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset by isolating entire battery degradation trajectories 
    for training and testing.

    Anti-Leakage Rationale:
        Applying standard random splits on time-series degradation data introduces 
        severe data leakage, as the model would 'see' future states of the same battery. 
        By strictly splitting based on unique battery IDs, we ensure the model is 
        evaluated on entirely unseen cells. This accurately simulates a real-world 
        deployment scenario where the BMS infers the health of a newly fielded battery.

    Args:
        all_df (pd.DataFrame): Combined DataFrame containing all battery cycles.
        train_ids (List[str]): List of battery IDs designated for the training set.
        test_ids (List[str]) : List of battery IDs designated for the testing set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The partitioned (df_train, df_test) DataFrames.
    """
    df_train = all_df[all_df["battery_id"].isin(train_ids)].copy()
    df_test  = all_df[all_df["battery_id"].isin(test_ids)].copy()

    print(f"  [Split] Train: {len(df_train):>4} cycles | Batteries: {train_ids}")
    print(f"  [Split] Test : {len(df_test):>4} cycles | Batteries: {test_ids}")

    return df_train, df_test


def get_feature_columns() -> List[str]:
    """
    Returns the strict list of feature columns used for model training.
    Explicitly excludes metadata identifiers ('battery_id') and target 
    labels ('soh', 'rul', 'is_eol') to prevent target leakage.
    
    Returns:
        List[str]: Selected numerical features.
    """
    return [
        "cycle_number",
        "voltage_drop",
        "voltage_mean",
        "voltage_std",
        "temp_max",
        "temp_mean",
        "temp_delta",
        "discharge_duration_s",
        "time_to_cutoff_s",
        "energy_wh",
        "coulombic_efficiency",
        "internal_resistance",
    ]

# =============================================================================
# 6. SCIKIT-LEARN PIPELINES: SCALER + RANDOM FOREST
# =============================================================================

def build_rf_pipeline(task: str = "soh") -> Pipeline:
    """
    Constructs a scikit-learn Pipeline integrating StandardScaler and RandomForestRegressor.

    Architectural Note:
        While feature scaling is not strictly required for Random Forests (as decision 
        trees are scale-invariant), it is included here as a best practice. This ensures 
        the pipeline remains seamlessly compatible with distance-based or gradient-descent 
        algorithms (e.g., Support Vector Regressors, Multilayer Perceptrons) in future 
        experimental iterations.

    Args:
        task (str): Target variable identifier ('soh' or 'rul'), utilized for 
                    descriptive naming and potential task-specific branching.

    Returns:
        Pipeline: An instantiated scikit-learn pipeline ready for fit/predict operations.
    """
    # Baseline hyperparameters for the Random Forest model
    rf_params = {
        "n_estimators"     : 200,      # Number of trees: balances variance reduction and computational cost
        "max_depth"        : 12,       # Constrains tree depth to prevent extreme overfitting
        "min_samples_split": 5,        # Minimum samples required to split an internal node
        "min_samples_leaf" : 2,        # Smooths predictions by enforcing a minimum leaf size
        "max_features"     : "sqrt",   # Subset of features considered per split (decorrelates trees)
        "n_jobs"           : -1,       # Utilizes all available CPU cores for parallel processing
        "random_state"     : 42,       # Ensures reproducibility across training runs
    }

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf",     RandomForestRegressor(**rf_params)),
    ], memory=None)

    return pipeline


# =============================================================================
# 7. FUTURE WORK - HYPERPARAMETER OPTIMIZATION (RandomizedSearchCV)
# =============================================================================

# def tune_hyperparameters(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series):
#     """
#     Future Implementation Note:
#     This module is reserved for automated Hyperparameter Tuning.
#     
#     It is currently bypassed to maintain pipeline simplicity and ensure full 
#     explainability of the baseline model mechanics during the research defense. 
#     
#     Future iterations will utilize cross-validation strategies (e.g., RandomizedSearchCV 
#     or GridSearchCV) to systematically explore the hyperparameter space (such as 
#     n_estimators, max_depth, and min_samples_split) to identify the optimal 
#     configuration that minimizes prediction error without overfitting.
#     """
#     pass
    
# =============================================================================
# 8. EVALUATION AND METRICS
# =============================================================================

def evaluate_model(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   task:   str = "SOH") -> Dict[str, float]:
    """
    Calculates and displays standard regression metrics for SOH or RUL estimation.

    Metrics Overview:
        - RMSE (Root Mean Square Error): Heavily penalizes large prediction errors, 
          which is crucial for safety-critical RUL predictions where overestimation 
          can lead to catastrophic failure.
        - MAE (Mean Absolute Error): Provides direct interpretability in the 
          target variable's native units (Ah percentage for SOH, cycles for RUL).
        - R² (Coefficient of Determination): Represents the proportion of variance 
          in the dependent variable that is predictable from the independent features.

    Args:
        y_true (np.ndarray): Ground truth target values.
        y_pred (np.ndarray): Model predicted values.
        task (str)         : Identifier for the prediction task ('SOH' or 'RUL') 
                             used for display formatting.

    Returns:
        Dict[str, float]: A dictionary containing the computed RMSE, MAE, and R² scores.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    unit = "" if task == "SOH" else " cycles"

    print(f"\n  ── Metrics [{task}] ──────────────────────")
    print(f"    RMSE : {rmse:.4f}{unit}")
    print(f"    MAE  : {mae:.4f}{unit}")
    print(f"    R²   : {r2:.4f}")
    print(f"  ─────────────────────────────────────────")

    return {"rmse": rmse, "mae": mae, "r2": r2}


# =============================================================================
# 9. VISUALIZATIONS
# =============================================================================

def plot_predictions(df_test: pd.DataFrame,
                     y_pred_soh: np.ndarray,
                     y_pred_rul: np.ndarray,
                     metrics_soh: Dict,
                     metrics_rul: Dict,
                     save_path: Optional[str]):
    """
    Generates a comprehensive visualization panel comparing predictions vs. 
    ground truth across degradation cycles.

    Generated Subplots:
        1. SOH: Actual vs. Predicted + EOL Threshold line (80%)
        2. RUL: Actual vs. Predicted across cycles
        3. SOH Residuals: Distribution of prediction errors
        4. RUL Residuals: Distribution of prediction errors

    Args:
        df_test     : Testing DataFrame containing 'cycle_number', 'soh', 'rul' columns.
        y_pred_soh  : Array of SOH predictions.
        y_pred_rul  : Array of RUL predictions.
        metrics_soh : Dictionary of evaluation metrics for the SOH model.
        metrics_rul : Dictionary of evaluation metrics for the RUL model.
        save_path   : Path to save the generated figure (None = display only).
    """
    cycles = df_test["cycle_number"].values

    # Publication-style color palette
    BLUE   = "#1A73E8"
    RED    = "#E8291A"
    GREEN  = "#1AE87A"
    ORANGE = "#E87A1A"
    GRAY   = "#AAAAAA"
    BG     = "#0D1117"
    PANEL  = "#161B22"

    fig = plt.figure(figsize=(12,7), facecolor=BG)
    fig.suptitle(
        "BMS Predictive Pipeline — SOH & RUL | Random Forest | NASA PCoE Dataset",
        fontsize=14, fontweight="bold", color="white", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── Plot 1: Actual vs. Predicted SOH ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(cycles, df_test["soh"].values, color=BLUE,   lw=2,   label="Actual SOH", zorder=3)
    ax1.plot(cycles, y_pred_soh,            color=RED,    lw=1.8, label="Predicted SOH",
             linestyle="--", zorder=4)
    ax1.axhline(y=SOH_EOL_THRESHOLD, color=ORANGE, lw=1.2, linestyle=":", alpha=0.8,
                label=f"EOL Threshold ({SOH_EOL_THRESHOLD:.0%})")
    ax1.fill_between(cycles, df_test["soh"].values, y_pred_soh, alpha=0.12, color=RED)
    ax1.set_title("State of Health (SOH)", color="white", fontsize=11, pad=8)
    ax1.set_xlabel("Discharge Cycle", color=GRAY, fontsize=9)
    ax1.set_ylabel("SOH (Normalized)", color=GRAY, fontsize=9)
    _style_ax(ax1, GRAY, BG)
    ax1.legend(fontsize=8, facecolor=PANEL, edgecolor=GRAY, labelcolor="white")
    ax1.text(0.97, 0.95,
             f"RMSE={metrics_soh['rmse']:.4f}\nMAE={metrics_soh['mae']:.4f}\nR²={metrics_soh['r2']:.4f}",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=8, color=GREEN,
             bbox=dict(boxstyle="round,pad=0.4", facecolor=BG, edgecolor=GREEN, alpha=0.8))

    # ── Plot 2: Actual vs. Predicted RUL ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(cycles, df_test["rul"].values, color=BLUE,   lw=2,   label="Actual RUL", zorder=3)
    ax2.plot(cycles, y_pred_rul,            color=RED,    lw=1.8, label="Predicted RUL",
             linestyle="--", zorder=4)
    ax2.fill_between(cycles, df_test["rul"].values, y_pred_rul, alpha=0.12, color=RED)
    ax2.set_title("Remaining Useful Life (RUL)", color="white", fontsize=11, pad=8)
    ax2.set_xlabel("Discharge Cycle", color=GRAY, fontsize=9)
    ax2.set_ylabel("RUL (Remaining Cycles)", color=GRAY, fontsize=9)
    _style_ax(ax2, GRAY, BG)
    ax2.legend(fontsize=8, facecolor=PANEL, edgecolor=GRAY, labelcolor="white")
    ax2.text(0.97, 0.95,
             f"RMSE={metrics_rul['rmse']:.2f}\nMAE={metrics_rul['mae']:.2f}\nR²={metrics_rul['r2']:.4f}",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=8, color=GREEN,
             bbox=dict(boxstyle="round,pad=0.4", facecolor=BG, edgecolor=GREEN, alpha=0.8))

    # ── Plot 3: SOH Residuals ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    residuals_soh = df_test["soh"].values - y_pred_soh
    ax3.scatter(cycles, residuals_soh, color=BLUE, alpha=0.6, s=18, zorder=3)
    ax3.axhline(0, color=ORANGE, lw=1.2, linestyle="--")
    ax3.fill_between(cycles, residuals_soh, 0, alpha=0.12, color=BLUE)
    ax3.set_title("SOH Residuals (Actual − Predicted)", color="white", fontsize=11, pad=8)
    ax3.set_xlabel("Discharge Cycle", color=GRAY, fontsize=9)
    ax3.set_ylabel("Residual Error", color=GRAY, fontsize=9)
    _style_ax(ax3, GRAY, BG)

    # ── Plot 4: RUL Residuals ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    residuals_rul = df_test["rul"].values - y_pred_rul
    ax4.scatter(cycles, residuals_rul, color=RED, alpha=0.6, s=18, zorder=3)
    ax4.axhline(0, color=ORANGE, lw=1.2, linestyle="--")
    ax4.fill_between(cycles, residuals_rul, 0, alpha=0.12, color=RED)
    ax4.set_title("RUL Residuals (Actual − Predicted)", color="white", fontsize=11, pad=8)
    ax4.set_xlabel("Discharge Cycle", color=GRAY, fontsize=9)
    ax4.set_ylabel("Residual Error (Cycles)", color=GRAY, fontsize=9)
    _style_ax(ax4, GRAY, BG)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"\n  [Plot] Prediction dashboard saved to: {save_path}")
    plt.show()


def plot_feature_importance(pipeline_soh: Pipeline,
                            pipeline_rul: Pipeline,
                            feature_names: List[str],
                            save_path: Optional[str] = "result/feature_importance.png"):
    """
    Plots the relative feature importances for both the SOH and RUL models.

    Utilizes the 'feature_importances_' attribute native to RandomForest (impurity-based).
    Note: For a more robust analysis in future work, permutation importance is recommended.

    Args:
        pipeline_soh  : Trained Pipeline for SOH prediction.
        pipeline_rul  : Trained Pipeline for RUL prediction.
        feature_names : List of feature column names.
        save_path     : Path to save the generated figure.
    """
    rf_soh = pipeline_soh.named_steps["rf"]
    rf_rul = pipeline_rul.named_steps["rf"]

    imp_soh = rf_soh.feature_importances_
    imp_rul = rf_rul.feature_importances_

    # Sort based on the average importance across both models
    idx = np.argsort((imp_soh + imp_rul) / 2)[::-1]

    BG    = "#0D1117"
    PANEL = "#161B22"
    BLUE  = "#1A73E8"
    RED   = "#E8291A"
    GRAY  = "#AAAAAA"

   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.suptitle("Feature Importance — Random Forest", color="white",
                 fontsize=13, fontweight="bold")

    for ax, imp, label, color in [
        (ax1, imp_soh, "SOH", BLUE),
        (ax2, imp_rul, "RUL", RED),
    ]:
        ax.set_facecolor(PANEL)
        bars = ax.barh([feature_names[i] for i in idx],
                       imp[idx], color=color, alpha=0.85)
        ax.set_title(f"Importance — {label}", color="white", fontsize=11)
        ax.set_xlabel("Importance (Gini Impurity)", color=GRAY, fontsize=9)
        _style_ax(ax, GRAY, BG)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  [Plot] Feature importance saved to: {save_path}")
    plt.show()


def _style_ax(ax, text_color, bg_color):
    ax.tick_params(colors=text_color)
    ax.spines["bottom"].set_color(text_color)
    ax.spines["left"].set_color(text_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(bg_color)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(text_color)

# =============================================================================
# 10. MAIN PIPELINE ORCHESTRATION
# =============================================================================

def run_pipeline(run_hpo: bool = False):
    """
    Orchestrates the end-to-end training and evaluation pipeline.

    Args:
        run_hpo (bool): If True, triggers hyperparameter optimization (RandomizedSearchCV) - (Feature marked as Future Work).
                        If False, utilizes the default hyperparameters defined in build_rf_pipeline().
    """
    print("=" * 65)
    print("  BMS Predictive Pipeline — SOH & RUL")
    print("  Model: RandomForestRegressor | scikit-learn")
    print("=" * 65)

    # ── Step 1: Data Loading & Parsing ────────────────────────────────────────
    print("\n[1/6] Loading data...")
    all_dfs = []

    print("  >> Parsing NASA PCoE .mat files...")
    for bid in BATTERY_IDS_TRAIN + BATTERY_IDS_TEST:

        base_path = Path(__file__).resolve().parent
        data_dir  = base_path / DATA_FOLDER_NAME
        mat_path  = data_dir / f"{bid}.mat"

        if not mat_path.exists():
            raise FileNotFoundError(
                f"File not found: {mat_path}\n"
                f"Please download the dataset from: https://www.kaggle.com/datasets/ckskaggle/li-ion-battery-dataset-from-nasa-pcoe/data"
            )
        
        raw  = load_battery_mat(str(mat_path))
        df_b = build_feature_dataframe(raw, bid)
        df_b = compute_soh_rul(df_b)
        all_dfs.append(df_b)
        print(f"    Loaded: {bid} → {len(df_b)} discharge cycles")

    all_df = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total cycles in dataset: {len(all_df)}")
    print(f"  Extracted columns: {list(all_df.columns)}")

    # ── Step 2: Battery-Level Split (Leakage Prevention) ──────────────────────
    print("\n[2/6] Splitting data (battery-wise isolated split)...")
    df_train, df_test = split_by_battery(all_df, BATTERY_IDS_TRAIN, BATTERY_IDS_TEST)

    feature_cols = get_feature_columns()

    # Drop any cycles containing NaN values in critical features or targets
    df_train = df_train.dropna(subset=feature_cols + ["soh", "rul"])
    df_test  = df_test.dropna(subset=feature_cols + ["soh", "rul"])

    X_train = df_train[feature_cols].values
    X_test  = df_test[feature_cols].values

    y_train_soh = df_train["soh"].values
    y_test_soh  = df_test["soh"].values
    y_train_rul = df_train["rul"].values
    y_test_rul  = df_test["rul"].values

    print(f"  Active Features: {feature_cols}")
    print(f"  X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")

    # ── Step 3: Pipeline Construction ─────────────────────────────────────────
    print("\n[3/6] Constructing Random Forest pipelines...")
    pipe_soh = build_rf_pipeline(task="soh")
    pipe_rul = build_rf_pipeline(task="rul")

    # ── Step 4: Model Training ────────────────────────────────────────────────
    print("\n[4/6] Training models...")
    if run_hpo:
        print("  >> Hyperparameter optimization requested. (Feature marked as Future Work)")
        print("  >> Falling back to default hyperparameters...")
        pipe_soh.fit(X_train, y_train_soh)
        pipe_rul.fit(X_train, y_train_rul)
    else:
        print("  >> Utilizing default hyperparameters (run_hpo=False)")
        pipe_soh.fit(X_train, y_train_soh)
        pipe_rul.fit(X_train, y_train_rul)

    print("  Models trained successfully.")

    # ── Step 5: Prediction & Evaluation ───────────────────────────────────────
    print("\n[5/6] Evaluating models on hold-out test set...")
    y_pred_soh = pipe_soh.predict(X_test)
    y_pred_rul = pipe_rul.predict(X_test)

    # Physics constraints: SOH ∈ [0,1], RUL ≥ 0
    y_pred_soh = np.clip(y_pred_soh, 0.0, 1.0)
    y_pred_rul = np.clip(y_pred_rul, 0.0, None)

    metrics_soh = evaluate_model(y_test_soh, y_pred_soh, task="SOH")
    metrics_rul = evaluate_model(y_test_rul, y_pred_rul, task="RUL")

    # ── Step 6: Visual Analytics ──────────────────────────────────────────────
    print("\n[6/6] Generating visual dashboards...")
    
    # Ensure the result directory exists
    result_folder = base_path / 'result'
    result_folder.mkdir(parents=True, exist_ok=True)

    plot_predictions(df_test, y_pred_soh, y_pred_rul,
                     metrics_soh, metrics_rul,
                     save_path=str(result_folder / "bms_predictions.png"))

    plot_feature_importance(pipe_soh, pipe_rul,
                            feature_names=feature_cols,
                            save_path=str(result_folder / "feature_importance.png"))

    print("\n" + "=" * 65)
    print("  Pipeline execution completed successfully.")
    print("=" * 65)

    return {
        "pipeline_soh"  : pipe_soh,
        "pipeline_rul"  : pipe_rul,
        "df_train"      : df_train,
        "df_test"       : df_test,
        "y_pred_soh"    : y_pred_soh,
        "y_pred_rul"    : y_pred_rul,
        "metrics_soh"   : metrics_soh,
        "metrics_rul"   : metrics_rul,
        "feature_cols"  : feature_cols,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    results = run_pipeline()