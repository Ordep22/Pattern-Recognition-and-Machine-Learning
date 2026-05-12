"""
Assignment 01: Math Vectorized Operations and Aggregation

Course: Pattern Recognition and Machine Learning - UTFPR

Objective: Classification using basic ML techniques on the Wine Dataset.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# ============================================================
# 1. DATA LOADING & PATH MANAGEMENT
# ============================================================

# STUDY NOTE: 
# Using Path(__file__) ensures the script finds the data folder 
# regardless of where the terminal is running from.
base_path  = Path(__file__).resolve().parent
data_dir = base_path / 'wine_data_set'

# Locating the .data file
file = [f for f in os.listdir(data_dir) if f.endswith('.data')]
file_path = data_dir / file[0]

# STUDY NOTE: header=None is mandatory for raw UCI datasets to avoid 
# treating the first sample as column names.
wine_df = pd.read_csv(file_path, header=None)
print(type(wine_df))
print(wine_df.head())


# ============================================================
# 2. FEATURE & TARGET SELECTION
# ============================================================

# In Wine dataset: Column 0 is the Class, Columns 1: are Features
X = wine_df.iloc[:, 1:]
y = wine_df.iloc[:, 0]

# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

# 70% Training / 30% Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# ============================================================
# 4. PREPROCESSING & MODEL TRAINING
# ============================================================

"""
STUDY NOTE ON PREPROCESSING:
Standardization is critical for the Perceptron algorithm (gradient-based).
We fit the scaler ONLY on the training set to prevent 'Data Leakage'
from the test set into the model's knowledge base.
"""
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Model Initialization and Fitting
model = Perceptron()
model.fit(X_train_std, y_train)


# ============================================================
# 5. PREDICTION & EVALUATION
# ============================================================

predictions = model.predict(X_test_std)

print("\n" + 30*"---")
print("CLASSIFICATION REPORT")
print(30*"---")
print(classification_report(y_test, predictions, target_names=['Class 1', 'Class 2', 'Class 3']))

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_true=y_test, y_pred=predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)

print(30*"---")
print("Output Confusion Matrix:")
print(conf_matrix)
print(30*"---")

disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Wine Classification")
plt.savefig("result/confusion_catrix_wine_classification.png")
plt.show()


