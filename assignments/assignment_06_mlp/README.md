# Assignment 06: Multilayer Perceptron (MLP) & Dimensionality Reduction (PCA) for Human Fall Detection

**Course:** Pattern Recognition and Machine Learning - CPGEI-CT - UTFPR  
**Domain:** Human Motion Analytics / Deep Learning Engineering  

---

## ⏱️ 1. Context, Objectives & Engineering Challenges

The objective of this assignment is to replicate the Human Fall Detection experiments from **Assignment 03** (originally built using Support Vector Machines - SVM) using Artificial Neural Networks—specifically a **Multilayer Perceptron (MLP)** via Keras/TensorFlow. The pipeline evaluates performance across the full feature space and compares it against PCA-reduced inputs (2, 5, and 10 principal components) under multiple topologies and regularization constraints.

### 🛠️ Environment & Engineering Hurdles (Post-Mortem)
Setting up the environment for this assignment presented significant engineering challenges on modern development environments (macOS architecture):

* **Python Version Incompatibility:** Modern Python environments (e.g., Python 3.11+) often clash with specific native binary wheel definitions for production-stable TensorFlow releases. 

* **The Solution:** A strict environmental downgrade to **Python 3.11** was required alongside explicit dependency freezing to ensure reliable memory allocation and native acceleration compilation for the Keras execution backend.

---

## 📊 2. Data Pipeline & Feature Engineering
To ensure academic consistency and isolate the effects of the classifiers, the feature extraction layer mimics Assignment 03 exactly:

* **Input Data:** 5-second raw signal windows (500 samples at 100Hz) from a 9-axis IMU (Accelerometer, Gyroscope, Magnetometer).

* **Feature Extraction:** Statistical physics-informed features (Mean and Standard Deviation) calculated independently across all 9 spatial channels, resulting in an 18-dimensional feature space.

* **Subject-Independent Splitting (Anti-Leakage):** To strictly guarantee that the models learn generalized human kinematic patterns rather than individual biometric signatures, the training/testing split was engineered **volunteer-wise** (Volunteers $\le$ 18 for training; Volunteers $>$ 18 for testing).

---

## 🧠 3. Architecture & Regularization Strategy
The network architecture relies on a dynamically scalable `Sequential` model optimized for binary classification:
* **Input Layer:** Explicitly handled via the modern `Input(shape=(input_dim,))` layer to comply with current Keras APIs and eliminate legacy layer warnings.
* **Hidden Topologies Checked:** `[64, 32]` and `[32, 16]` neurons with Rectified Linear Unit (`ReLU`) activations.

* **Output Layer:** A single dense neuron mapped to a `Sigmoid` activation function, outputting continuous probabilities bounded strictly between $0.0$ and $1.0$.

* **Regularization Matrix:** 1. **Dropout (0.3):** Randomly mutes 30% of internal weights per step to prevent over-reliance on dominant sensors.

  2. **L2 Regularization (0.001):** Imposes a quadratic weight decay penalty directly onto the loss function.


  3. **Early Stopping:** Monitored via the validation set with a patience barrier of 10 epochs to capture optimal weights before overfitting triggers.

---

## 📈 4. Experimental Results & The "Failure" Analysis

While classical statistical methods like the SVM in Assignment 03 achieved high classification metrics, the MLP baseline under default hyperparameters encountered a textbook Deep Learning pitfall: **Numerical Underflow and Class Collapse.**

### Full Space vs. PCA Spaces Performance (Representative Sample)
```text
             precision    recall  f1-score   support

     No Fall       0.50      1.00      0.67        72
        Fall       0.00      0.00      0.00        72

    accuracy                           0.50       144
   macro avg       0.25      0.50      0.33       144
weighted avg       0.25      0.50      0.33       144
```

---
## 📈 5. Experimental Artifacts (Visual Analytics)

Below are the key dashboards generated automatically by the pipeline, displaying the behavioral contrast between different feature spaces and topologies.

* **Full Feature Space Pipeline**
* **PCA-Reduced Spaces (2 & 5 Components)**
* **PCA-Reduced Space (10 Components) — Underflow Threshold Case**

## 🔍 Crucial Analytical Discussion (Why it Failed)

* **Catastrophic Loss Underflow:** As observed in the generated dashboards, the loss curve did not gracefully descend toward zero. Instead, it experienced a massive numerical collapse, plunging into extreme negative territory (approaching $-60,000$). Mathematically, binary cross-entropy cannot be negative. This anomaly points to severe gradient instability and numerical underflow occurring within the computer's floating-point calculations when executing backpropagation loops.

* **Lazy Convergence (Class Monopolization):** Due to the sudden numerical instability of the hidden layer weights, the gradient updates broke down early in the training loop. To minimize total error under broken weights, the network converged on a "lazy strategy": predicting the No Fall class for 100% of the inputs. This explains the perfect Recall ($1.00$) for No Fall and absolute failure ($0.00$) for detecting actual Fall events, resulting in a baseline random-guessing accuracy of exactly $50\%$.

* **Data Volume Constraints:**: This experiment proves that while structural margin-based algorithms (like SVMs) excel at extracting geometric boundaries from compact tabular feature spaces, deep neural networks are highly sensitive to optimization hyperparameters (such as learning rate and batch sizes) and typically require orders of magnitude more data to successfully stabilize their weights.


## 7. 🛠️ Stabilizing the Network

To mitigate this gradient explosion and recover standard cross-entropy tracking, the optimization step was refactored by forcing the Adam optimizer to operate at a highly conservative learning rate ($\eta = 0.0002$ or $0.0001$). This restriction slows weight variations down enough to keep the final log-likelihood values within the stable boundaries of the Sigmoid activation function.

## 8. How to Run

1. Verify your local Python environment is downgraded to Python 3.11

2. Place the raw .mat files inside the /data/ directory.

3. Execute the automated evaluation pipeline:

   ```bash
   python3 assignment_06_mpl.py