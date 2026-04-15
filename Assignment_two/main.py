# Import essential libraries for data manipulation, visualization, and machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load the Titanic dataset (Training and Testing sets)
df_gender_submission  = pd.read_csv(r"/Users/pedropereira/Documents/Work/Pattern-Recognition-and-Machine-Learning/Assignment_two/titanic/gender_submission.csv")
df_test = pd.read_csv(r"/Users/pedropereira/Documents/Work/Pattern-Recognition-and-Machine-Learning/Assignment_two/titanic/test.csv")
df_train = pd.read_csv(r"/Users/pedropereira/Documents/Work/Pattern-Recognition-and-Machine-Learning/Assignment_two/titanic/train.csv")


# Feature Selection: Dropping columns that do not 
# provide significant predictive power (high cardinality or noise)
df_gender_submission.dropna(inplace = True)
df_test.drop(columns = ['Name', 'Ticket','Cabin'], inplace = True)
df_train.drop(columns = ['Name', 'Ticket', 'Cabin'], inplace = True)


# Data Imputation: Handling missing values using 
# mean and median strategies to preserve data distribution
median_age  = df_train['Age'].mean()
df_test['Age'] = df_test['Age'].fillna(median_age)
df_train['Age'] = df_train['Age'].fillna(median_age)

median_fare_test = df_test['Fare'].mean()
df_test['Fare'] = df_test['Fare'].fillna(median_fare_test)

median_fare_train = df_train['Fare'].median()
df_train['Fare'] = df_train['Fare'].fillna(median_fare_train)

# Categorical Encoding: Mapping string labels 
# to numerical values for model compatibility
df_test['Sex'] = df_test['Sex'].map({'male': 0.0 , 'female': 1.0})
df_train['Sex'] = df_train['Sex'].map({'male': 0.0 , 'female': 1.0})

df_test['Embarked'] = df_test['Embarked'].fillna('S')
df_test['Embarked'] = df_test['Embarked'].map({'S':0.0,'C':1.0,'Q':2.0})

df_train['Embarked'] = df_train['Embarked'].fillna('S')
df_train['Embarked'] = df_train['Embarked'].map({'S':0.0,'C':1.0,'Q':2.0})


# Model Initialization: Training a 
# Logistic Regression classifier on the scaled training data
scaler = StandardScaler()

X_train = df_train.drop(columns=['Survived','PassengerId'])
X_test = df_test.drop(columns=['PassengerId'])
Y_train = df_train['Survived']

# Re-fit the scaler only on selected features to ensure consistent dimensionality
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train dataset: {df_train.isnull().sum()}")
print(f"Test dataset: {df_test.isnull().sum()}")


# Train the classifier using the final processed feature matrix
model = LogisticRegression()
model.fit(X_train_scaled,Y_train)

#Predict result
predictions  = model.predict(X_train_scaled)

#Generate the propablistic 
propablistic = model.predict_proba(X_train_scaled)

print(30*"---"+"\n")
print("Gender Submission\n")
print(df_gender_submission)
print(30*"---"+"\n")

print(30*"---"+"\n")
print("DF test\n")
print(df_test)
print(30*"---"+"\n")

print(30*"---"+"\n")
print("DF train\n")
print(df_train)
print(30*"---"+"\n")

print(30*"---"+"\n")
print("Prediction\n")
print(predictions)
print(30*"---"+"\n")

print(30*"---"+"\n")
print("Prediction\n")
print(propablistic)
print(30*"---"+"\n")

# performance metrics (Confusion Matrix and Classification Report)
conf_matrix = confusion_matrix(Y_train, predictions)
print(30*"---"+"\n")
print("Confusion Matrix:")
print(conf_matrix)
print(30*"---"+"\n")

# Model Evaluation: Generating predictions and computing 
print(30*"---"+"\n")
print("Prediction\n")
print(classification_report(Y_train, predictions,target_names=['Died', 'Suvived']))
print(30*"---"+"\n")
