"""
Assignment 02: Implement Scikit-Learn Tools to predict survivors on the Titanic dataset

Course: Pattern Recognition and Machine Learning
Objective: Predict whether or not a passenger survived the Titanic disaster using Logistic Regression.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ---- Domain Global Parameters ----
DEBUG = False

# ============================================================
# 1. LOAD THE TITANIC DATASET
# ============================================================

base_path = Path(__file__).resolve().parent
data_dir = base_path / 'data'

train_file_path         = data_dir / 'train.csv'
test_file_path          = data_dir / 'test.csv'
gender_submission_path  = data_dir / 'gender_submission.csv'

df_train             = pd.read_csv(train_file_path)
df_test              = pd.read_csv(test_file_path)
df_gender_submission = pd.read_csv(gender_submission_path)

# Extract Title from Name
df_train['Title'] = df_train['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
df_test['Title']  = df_test['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
# ============================================================
# 2. Feature Selection:
# ============================================================

# 2.1 Dropping columns lacking significant predictive power
df_gender_submission.dropna(inplace = True)
df_test.drop(columns = ['Name', 'Ticket','Cabin'], inplace = True)
df_train.drop(columns = ['Name', 'Ticket', 'Cabin'], inplace = True)


# 2.2 Data Imputation (Preventing Data Leakage)
# Age
median_age  = df_train['Age'].mean()
df_test['Age'] = df_test['Age'].fillna(median_age)
df_train['Age'] = df_train['Age'].fillna(median_age)

# Fare
median_fare_test = df_test['Fare'].mean()
df_test['Fare'] = df_test['Fare'].fillna(median_fare_test)

median_fare_train = df_train['Fare'].median()
df_train['Fare'] = df_train['Fare'].fillna(median_fare_train)

# 2.3 Categorical Encoding
df_test['Sex'] = df_test['Sex'].map({'male': 0.0 , 'female': 1.0})
df_train['Sex'] = df_train['Sex'].map({'male': 0.0 , 'female': 1.0})

df_test['Embarked'] = df_test['Embarked'].fillna('S')
df_test['Embarked'] = df_test['Embarked'].map({'S':0.0,'C':1.0,'Q':2.0})

df_train['Embarked'] = df_train['Embarked'].fillna('S')
df_train['Embarked'] = df_train['Embarked'].map({'S':0.0,'C':1.0,'Q':2.0})

# 2.4 Family Size Calculation
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1
median_family_size_test  = df_test['FamilySize'].mean()
df_test['FamilySize'] = df_test['FamilySize'].fillna(median_family_size_test)

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
median_family_size_train  = df_train['FamilySize'].mean()
df_train['FamilySize'] = df_train['FamilySize'].fillna(median_family_size_train)

# 2.5 Title Mapping
title_mapping = {
    "Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, 
    "Col": 4, "Dr": 4, "Rev": 4, "Major": 4, "Mlle": 1, "Ms": 1, "Mme": 2, 
    "Lady": 4, "Countess": 4, "Sir": 4, "Don": 4, "Dona": 4, "Jonkheer": 4, "Capt": 4
}

# Apply the mapping
df_train['Title'] = df_train['Title'].map(title_mapping)
df_test['Title'] = df_test['Title'].map(title_mapping)

# IMPORTANT: Fallback for any unmapped titles (fills remaining NaNs)
mode_title = df_train['Title'].mode()[0]
df_train['Title'] = df_train['Title'].fillna(mode_title)
df_test['Title'] = df_test['Title'].fillna(mode_title)


# ============================================================
# 3. MODEL INITIALIZATION & TRAINING
# ============================================================

# Separate features and targets
X_train = df_train.drop(columns=['Survived', 'PassengerId'])
y_train = df_train['Survived']

X_test = df_test.drop(columns=['PassengerId'])
# Extract actual ground truth from gender_submission for evaluation
y_test = df_gender_submission['Survived'] 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train the Logistic Regression classifier
model = LogisticRegression(C=0.05, random_state=42)
model.fit(X_train_scaled, y_train)


# ============================================================
# 4. Predict result
# ============================================================

predictions = model.predict(X_test_scaled)

if DEBUG:
    print(30*"-"+"\nPredictions\n")
    print(predictions[:10])


# ============================================================
# 5. EVALUATION METRICS
# ============================================================
confmat = confusion_matrix(y_true=y_test, y_pred=predictions)

print("\n" + 30*"-")
print("CLASSIFICATION REPORT (TEST SET)")
print(30*"-")
print(classification_report(y_test, predictions, target_names=['Died', 'Survived']))

# ============================================================
# 5.1 VISUALIZATION & OUTPUT
# ============================================================
result_folder = base_path / 'result' 
result_folder.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center', fontsize=12)

plt.title('Confusion Matrix (Test Set)', pad=15)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(result_folder / 'confusion_matrix.png', dpi=150)
plt.show()
