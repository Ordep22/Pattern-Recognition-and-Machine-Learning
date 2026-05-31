# Pattern Recognition (RPD-0041)
### CPGEI-CT - UTFPR | Professor: André E. Lazzaretti

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

This repository contains the practical assignments and study materials developed during the **Pattern Recognition** Master's course at the Federal University of Technology – Paraná (UTFPR). The project focuses on building end-to-end Machine Learning pipelines, including feature extraction from raw sensors, dimensionality reduction, and rigorous model evaluation.

## 📂 Repository Structure

### 1. Lab Assignments (`/assignments`)
Official course deliverables focused on real-world and synthetic data challenges.

* **Assignment 01 - Implementation of the Perceptron & ADALINE:** 
    * Classification methods applied to the Wine Dataset to achieve optimized performance.
    * **Key Concepts:** Perceptron algorithm (a foundational model in academia), ADALINE (Adaptive Linear Neuron) implementation from scratch, and Gradient Descent optimization.
* **Assignment 03 - Human Fall Detection:**
    * Development of a classification system using IMU sensor data (Accelerometer, Gyroscope, Magnetometer).
    * **Key Concepts:** SVM, Logistic Regression, Feature Engineering, Precision/Recall trade-offs, and Confusion Matrix analysis.
* **Assignment 04 - Dimensionality Reduction & Projections:**
    * Comparison between linear and non-linear compression techniques.
    * **Key Concepts:** Linear Discriminant Analysis (LDA), Principal Component Analysis (PCA), and Kernel PCA (RBF kernel) applied to 3D Gaussian distributions and Archimedean spirals.
* **Assignment 05 - Clustering & Data Segmentation:**
    * Autonomous determination of the optimal number of clusters for unlabelled synthetic datasets.
    * **Key Concepts:** K-Means++, Hierarchical Clustering, Elbow Method, and Silhouette Coefficient.
* **Final Project - BMS Predictive Pipeline (SOH & RUL):**
    * **Course Objective:** Build a data-driven regression pipeline to predict State of Health (SOH) and Remaining Useful Life (RUL) using classical Machine Learning.
    * **Research Objective:** Serve as a foundational proof-of-concept for ongoing Master's dissertation research in Applied Computing. This pipeline generates the necessary technical artifacts to critically evaluate whether the Random Forest algorithm is a viable and efficient baseline strategy for battery health prognostics within a Smart BMS architecture.

### 2. Book Exercises (`/python_ml_book`)
Implementations and notes based on *Python Machine Learning* by Sebastian Raschka.
* Fundamental algorithms from scratch: Perceptron, Adaline, and Decision Trees.
* In-depth study of Information Gain metrics (Gini Impurity vs. Entropy) and Ensemble methods like Random Forests.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Data Processing:** NumPy, Pandas, SciPy
* **Machine Learning:** Scikit-Learn
* **Visualization:** Matplotlib (2D/3D), Seaborn

## 🔬 Methodology
To ensure academic rigor and prevent model bias, the following practices were implemented across the projects:
1. **Strict Data Leakage Prevention:** Training/Testing splits were carefully designed around unique entities (e.g., splitting by Volunteer ID in fall detection, and by Battery ID in BMS prognostics) to ensure models generalize accurately to entirely unseen subjects or hardware in real-world deployments.
2. **Data Standardization:** Application of `StandardScaler` to prevent feature dominance in distance-based algorithms.
3. **Hyperparameter Tuning:** Use of `GridSearchCV` and Cross-Validation to optimize model parameters.

---

## 🚀 How to Run
1. Clone the [repository](https://github.com/Ordep22/Pattern-Recognition-and-Machine-Learning):

   ```bash
   git clone https://github.com/Ordep22/Pattern-Recognition-and-Machine-Learning.git