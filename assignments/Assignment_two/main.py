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
df_gender_submission  = pd.read_csv(r"/Users/pedropereira/Documents/" \
"Work/Pattern-Recognition-and-Machine-Learning/Assignment_two/titanic/gender_submission.csv")
df_test = pd.read_csv(r"/Users/pedropereira/Documents/" \
"Work/Pattern-Recognition-and-Machine-Learning/Assignment_two/titanic/test.csv")
df_train = pd.read_csv(r"/Users/pedropereira/Documents/" \
"Work/Pattern-Recognition-and-Machine-Learning/Assignment_two/titanic/train.csv")

df_test['Title'] = df_test['Name'].str.extract(' ([A-Za-z]+)\\.', expand = False)
print(df_test['Title'])
df_train['Title'] = df_train['Name'].str.extract(' ([A-Za-z]+)\\.', expand = False)
print(df_train.head())


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


df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1
median_family_size_test  = df_test['FamilySize'].mean()
df_test['FamilySize'] = df_test['FamilySize'].fillna(median_family_size_test)


df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
median_family_size_train  = df_train['FamilySize'].mean()
df_train['FamilySize'] = df_train['FamilySize'].fillna(median_family_size_train)


title_mapping = {
    "Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, 
    "Col": 4, "Dr": 4, "Rev": 4, "Major": 4, "Mlle": 1, "Ms": 1, "Mme": 2, 
    "Lady": 4, "Countess": 4, "Sir": 4, "Don": 4, "Dona": 4, "Jonkheer": 4, "Capt": 4
}

# Apply the mapping
df_train['Title'] = df_train['Title'].map(title_mapping)
df_test['Title'] = df_test['Title'].map(title_mapping)


mode_title = df_train['Title'].mode()[0]
df_train['Title'] = df_train['Title'].fillna(mode_title)
df_test['Title'] = df_test['Title'].fillna(mode_title)


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
model = LogisticRegression(C = 0.05)
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
confmat = confusion_matrix(y_true = Y_train, y_pred = predictions)
print(30*"---")
print("Output Confusion Matrix")
print(confmat)
print(30*"---")


fix, ax  = plt.subplots(figsize  = (2.5, 2.5))
ax.matshow(confmat, cmap = plt.cm.Blues, alpha = 0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x = j, y = i, s = confmat[i, j], va = 'center', ha = 'center')

plt.xlabel('prediction')
plt.ylabel('True label')
plt.show()


# Model Evaluation: Generating predictions and computing 
print(30*"---"+"\n")
print("Prediction\n")
print(classification_report(Y_train, predictions,target_names=['Died', 'Suvived']))
print(30*"---"+"\n")
