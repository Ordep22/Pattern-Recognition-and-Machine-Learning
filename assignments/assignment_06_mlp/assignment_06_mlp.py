"""
Assignment 06: MLP Classifier & Dimensionality Reduction (PCA) for Human Fall Detection

Course: Pattern Recognition and Machine Learning - CPGEI-CT - UTFPR
Objective: Implement Multilayer Perceptrons (MLPs) using Keras to classify human falls,
           exploring structural topologies, PCA configurations, and regularization features.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io as wavefile
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC, Precision, Recall

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

# ----- Global Configuration Parameters ----
DEBUG = False
REANDON_STATE = 42

# =============================================================================
# 1.0 INFRASTRUCTURE: PATH & DIRECTORY VERIFICATION
# =============================================================================
rows_test = []
rows_train = []

root_path = Path(__file__).resolve().parent
data_dir = root_path / 'data'

result_folder = root_path / 'result'
result_folder.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2.0 DATA LOADING & FEATURE EXTRACTION
# =============================================================================
print("\n" + 65*"=")
print("  Loading Dataset and Running Subject-Independent Split...")
print(65*"=")

for i in os.listdir(data_dir):
    if i.startswith('.') or not i.endswith('.mat'):
        continue

    mat = wavefile.loadmat(f"{data_dir}/{i}")
    filename = i
    parts = filename.split('.') 
    volunteer_id = int(parts[3])
    label = int(parts[0])  # 1 = Fall, 0 = No Fall

    signals = mat['newData']

    features = {
        'label': label,
        'volunteer': volunteer_id,
        'acc_x_mean': np.mean(signals[:, 1]),
        'acc_y_mean': np.mean(signals[:, 2]),
        'acc_z_mean': np.mean(signals[:, 3]),
        'acc_x_std': np.std(signals[:, 1]),
        'acc_y_std': np.std(signals[:, 2]),
        'acc_z_std': np.std(signals[:, 3]),

        'gyro_x_mean': np.mean(signals[:, 4]),
        'gyro_y_mean': np.mean(signals[:, 5]),
        'gyro_z_mean': np.mean(signals[:, 6]),
        'gyro_x_std': np.std(signals[:, 4]),
        'gyro_y_std': np.std(signals[:, 5]),
        'gyro_z_std': np.std(signals[:, 6]),

        'mag_x_mean': np.mean(signals[:, 7]),
        'mag_y_mean': np.mean(signals[:, 8]),
        'mag_z_mean': np.mean(signals[:, 9]),
        'mag_x_std': np.std(signals[:, 7]),
        'mag_y_std': np.std(signals[:, 8]),
        'mag_z_std': np.std(signals[:, 9])
    }

    if volunteer_id <= 18:
        rows_train.append(features)
    else:
        rows_test.append(features)

df_train = pd.DataFrame(rows_train)
df_test = pd.DataFrame(rows_test)

# =============================================================================
# 3.0 PREPROCESSING PIPELINE
# =============================================================================
X_train_raw = df_train.drop(columns=['label', 'volunteer']) 
Y_train = df_train['label'].values

X_test_raw = df_test.drop(columns=['label', 'volunteer']) 
Y_test = df_test['label'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# =============================================================================
# 4.0 DYNAMIC MLP ARCHITECTURE BUILDER
# =============================================================================
def build_mlp_model(input_dim: int, 
                    hidden_layers: list, 
                    dropout_rate: float = 0.3, 
                    l2_penalty: float = 0.001):
    """
    Constructs a Keras Sequential MLP with flexible topologies and regularizations.
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    # First Hidden Layer
    model.add(Dense(
        units=hidden_layers[0], 
        activation='relu',
        kernel_regularizer=l2(l2_penalty) if l2_penalty > 0 else None
    ))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
        
    # Dynamic Stacking of Additional Hidden Layers
    for units in hidden_layers[1:]:
        model.add(Dense(
            units=units, 
            activation='relu',
            kernel_regularizer=l2(l2_penalty) if l2_penalty > 0 else None
        ))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
            
    # Output Layer (Sigmoid for Binary Classification)
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy', 
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), 
        metrics=[AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
    )
    return model

# =============================================================================
# 5.0 EXPERIMENT RUNNER & EVALUATION VISUALS
# =============================================================================
def execute_experiment(X_tr, Y_tr, X_te, Y_te, topology, label_name):
    print("\n" + 50*"-")
    print(f" EXPERIMENT: {label_name} | Topology: {topology}")
    print(50*"-")
    
    input_shape_dim = X_tr.shape[1]
    model = build_mlp_model(input_dim=input_shape_dim, hidden_layers=topology)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_tr, Y_tr, 
                        validation_data=(X_te, Y_te), 
                        shuffle=False,
                        epochs=100, 
                        batch_size=16,
                        callbacks=[early_stop],
                        verbose=0)
    
    probabilities = model.predict(X_te, verbose=0)
    predictions = (probabilities >= 0.5).astype(int)
    
    print(classification_report(Y_te, predictions, target_names=['No Fall', 'Fall']))

    BLUE   = "#1A73E8"
    RED    = "#E8291A"
    GRAY   = "#AAAAAA"
    BG     = "#0D1117"

    # Evaluation Dashboard Generation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=BG)
    
    # 1. Loss Curves
    ax1.set_facecolor(BG)
    ax1.plot(history.history['loss'], color=BLUE, linewidth=2, label='Train Loss')
    ax1.plot(history.history['val_loss'], color=RED, linewidth=2, label='Val Loss')
    ax1.set_title(f'Loss Curve - {label_name}', color='white', fontsize=12, pad=10)
    ax1.set_xlabel('Epochs', color=GRAY)
    ax1.set_ylabel('Loss', color=GRAY)
    ax1.tick_params(colors=GRAY)
    ax1.legend(loc='upper right')
    ax1.grid(True, color='#21262D', linestyle='--')
    
    # 2. Confusion Matrix
    ax2.set_facecolor(BG)
    cm = confusion_matrix(Y_te, predictions)
    ax2.imshow(cm, cmap=plt.cm.Blues, alpha=0.6)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(x=j, y=i, s=cm[i, j], va='center', ha='center', 
                     fontsize=14, color='white' if cm[i, j] > cm.max()/2 else 'black')

    ax2.set_title('Confusion Matrix', color='white', fontsize=12, pad=10)
    ax2.set_xlabel('Predicted', color=GRAY)
    ax2.set_ylabel('True', color=GRAY)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['No Fall', 'Fall'], color=GRAY)
    ax2.set_yticklabels(['No Fall', 'Fall'], color=GRAY)
    ax2.tick_params(colors=GRAY)
    
    plt.tight_layout()
    filename_png = f"{result_folder}/{label_name.replace(' ', '_')}_{topology}.png"
    plt.savefig(filename_png, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
    plt.close()

# =============================================================================
# 6.0 AUTOMATED EXPERIMENT EXECUTION LOOP
# =============================================================================
topologies_to_test = [[64, 32], [32, 16]]

# 1. Experiments on Full Feature Space
for topo in topologies_to_test:
    execute_experiment(X_train_scaled, Y_train, X_test_scaled, Y_test, 
                       topology=topo, label_name="Full Feature Space")

# 2. Experiments on PCA-Reduced Spaces (2, 5, and 10 components)
pca_settings = [2, 5, 10]

for n_comp in pca_settings:
    # Fixed: Passing raw data here since the Pipeline already contains the StandardScaler steps
    preproc = SkPipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_comp, random_state=REANDON_STATE))
    ])
    X_train_pca = preproc.fit_transform(X_train_raw)  
    X_test_pca  = preproc.transform(X_test_raw) 
    
    for topo in topologies_to_test:
        execute_experiment(X_train_pca, Y_train, X_test_pca, Y_test, 
                           topology=topo, label_name=f"PCA Space ({n_comp} Components)")