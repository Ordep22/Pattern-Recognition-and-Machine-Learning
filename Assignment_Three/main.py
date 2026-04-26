import os
import numpy as np
import pandas as pd
import scipy.io as  wavefile
import matplotlib.pyplot as plt
from  sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# 1. DATA LOADING AND PREPARATION
# Load .mat files using scipy.io.loadmat [cite: 489]
# Split data by Volunteer ID: Train (01-18) vs Test (19-22) [cite: 498]

rows_test = []
rows_train = []

location = os.path.dirname(os.path.realpath(__file__))
content_path  = location + "/content"
print(f"Current Working Directory: {content_path}")

print(30*"---")
print("Loading Data  - split between train and test")
print(30*"---")

for i in os.listdir(content_path):

    mat = wavefile.loadmat(f"{content_path}/{i}")
    filename  = i
    parts = filename.split('.') 
    volunteer_id = int(parts[3])
    label = int(parts[0]) # To identify is it`s a fall or no fall

    signals = mat['newData']

    features = {
        'label': label,
        'volunteer': volunteer_id,
        'acc_x_mean': np.mean(signals[:, 1]),
        'acc_y_mean': np.mean(signals[:, 2]),
        'acc_z_mean': np.mean(signals[:, 3]),
        'gyro_x_mean': np.mean(signals[:, 4]),
        'gyro_y_mean': np.mean(signals[:, 5]),
        'gyro_z_mean': np.mean(signals[:, 6]),
        'mag_x_mean': np.mean(signals[:, 7]),
        'mag_y_mean': np.mean(signals[:, 8]),
        'mag_z_mean': np.mean(signals[:, 9])
    }

    # Separate according to volunteer ID
    if volunteer_id <= 18:
        rows_train.append(features)
    else:
        rows_test.append(features)

    print(f"File {filename} - Loaded")


df_train = pd.DataFrame(rows_train)
print(30*"---")
print("---- Df Train ----")
print(df_train.head())
print(30*"---\n")

df_test = pd.DataFrame(rows_test)
print(30*"---")
print("---- Df Train ----")
print(df_test.head())
print(30*"---\n")
    
print(30*"---")
print("Load Data Finished")
print(30*"---\n")


# 2. FEATURE EXTRACTION FUNCTION
# Input: 5-second signal window (500 samples at 100Hz) [cite: 488]
# Process: Calculate metrics for each of the 9 sensors (Acc, Gyro, Mag) [cite: 483, 502]
# Suggested: Mean, Std Dev, Max, Min, Zero-Crossing Rate 





# 3. DATABASE CONSTRUCTION
# Create X_train, y_train and X_test, y_test
# Each row = 1 acquisition file; Each column = 1 extracted feature 

# 4. PREPROCESSING (PIPELINE)
# Standardization: Essential for SVM and Logistic Regression [cite: 43, 47]

# 5. MODEL SELECTION AND HYPERPARAMETER TUNING
# Use GridSearchCV with Cross-Validation [cite: 209, 224, 504]
# Models to compare: LogisticRegression and SVC (SVM)

# 6. MODEL EVALUATION
# Plot Learning Curves to diagnose Bias/Variance (Overfitting vs Underfitting) [cite: 50, 252, 504]
# Confusion Matrix and Classification Report [cite: 361, 384]