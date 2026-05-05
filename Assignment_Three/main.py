import os
import numpy as np
import pandas as pd
import scipy.io as  wavefile
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#DEFINES VARIABLES
DEBUG = False


# 1. DATA LOADING AND PREPARATION
# Load .mat files using scipy.io.loadmat
# Split data by Volunteer ID: Train (01-18) vs Test (19-22)

rows_test = []
rows_train = []

location = os.path.dirname(os.path.realpath(__file__))
content_path  = location + "/content"
print(f"Current Working Directory: {content_path}")

print(30*"---")
print("Loading Data  - split between train and test")
print(30*"---")


# 2. FEATURE EXTRACTION FUNCTION
# Input: 5-second signal window (500 samples at 100Hz) 
# Process: Calculate metrics for each of the 9 sensors (Acc, Gyro, Mag) 
# Suggested: Mean, Std Dev
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

    # Separate according to volunteer ID
    if volunteer_id <= 18:
        rows_train.append(features)
    else:
        rows_test.append(features)


df_train = pd.DataFrame(rows_train)

if DEBUG:
    print(30*"---")
    print("---- Df Train ----")
    print(df_train.head())
    print(30*"---")

df_test = pd.DataFrame(rows_test)

if DEBUG:
    print(30*"---")
    print("---- Df Test ----")
    print(df_test.head())
    print(30*"---")

if DEBUG:    
    print(30*"---")
    print("Load Data Finished")
    print(30*"---")

# 4. PREPROCESSING (PIPELINE)
# Standardization: Essential for SVM and Logistic Regression

'''
Get all dataset. Less the columns label and volunteers. 
Because it'll the informations that we'd like to figure out
'''
X_train  = df_train.drop(columns=['label', 'volunteer']) 
Y_train = df_train['label']

X_test  = df_test.drop(columns=['label', 'volunteer']) 
Y_test = df_test['label']

# 5. MODEL SELECTION AND HYPERPARAMETER TUNING
# Use GridSearchCV with Cross-Validation [cite: 209, 224, 504]
# Models to compare: LogisticRegression and SVC (SVM)

svm = svm.SVC(kernel="linear", C=1)
svm.fit(X_train, Y_train)

# 6. MODEL EVALUATION
# Plot Learning Curves to diagnose Bias/Variance (Overfitting vs Underfitting) [cite: 50, 252, 504]
# Confusion Matrix and Classification Report [cite: 361, 384]
predictions  = svm.predict(X_test)

if DEBUG:  
    print(30*"---"+"\n")
    print("Prediction\n")
    print(predictions)
    print(30*"---"+"\n")

# Model Evaluation: Generating predictions and computing 
print(30*"---"+"\n")
print("Prediction\n")
print(classification_report(Y_test, predictions,target_names=['Non-Fall', 'Fall']))
print(30*"---"+"\n")


# performance metrics (Confusion Matrix and Classification Report)
confmatrix = confusion_matrix(y_true = Y_test, y_pred = predictions)
disp = ConfusionMatrixDisplay(confusion_matrix = confmatrix, display_labels= svm.classes_)
print(30*"---")
print("Output Confusion Matrix")
print(confmatrix)
print(30*"---")

disp.plot()
plt.show()