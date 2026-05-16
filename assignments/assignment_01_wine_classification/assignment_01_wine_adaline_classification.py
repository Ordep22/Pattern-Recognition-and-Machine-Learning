"""
Assignment 01: Implementation of the ADALINE Classification Method for the Wine Dataset

Course: Pattern Recognition and Machine Learning - UTFPR

Objective: Perform classification using the ADALINE (Adaptive Linear Neuron) technique 
           to achieve optimized performance on the Wine dataset.
"""


import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. ADALINE Classe
# ============================================================

class AdalineGD:
    """Adaptive Linear Neuron classifier with Gradient Descent.
    
    Parameters:
    -----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for weight initialization.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data."""
        rgen = np.random.RandomState(self.random_state)
        # Initialize weights and bias
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)
        self.cost_ = []

        for _ in range(self.n_iter):
            # Net input and linear activation
            output = self.activation(self.net_input(X))
            errors = (y - output)
            
            # Update weights: w = w + eta * X.T.dot(errors)
            self.w_ += self.eta * X.T.dot(errors)
            self.b_ += self.eta * errors.sum()
            
            # Calculate Sum of Squared Errors (SSE)
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input: z = w·x + b"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation (identity function)"""
        return X

    def predict(self, X):
        """Return class label after unit step (threshold function)"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)




# ============================================================
# 2. DATA LOADING & PATH MANAGEMENT
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


# ============================================================
# 3. FEATURE & TARGET SELECTION
# ============================================================

wine_df_two_classes = wine_df[wine_df[0] != 3].copy()

wine_df_two_classes[0] = wine_df_two_classes[0].map({1:-1,2:1})

print("\n" + 30*"---")
print("DATA SET")
print(30*"---")
print(wine_df_two_classes.head(-5))
print(30*"---")


# In Wine dataset: Column 0 is the Class, Columns 1: are Features
#For all features
X = wine_df_two_classes.iloc[:, 1:]

#Select Alcohol and Ash
#X = wine_df_two_classes.iloc[:, [1,3]].values

y = wine_df_two_classes.iloc[:, 0]

# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

# 70% Training / 30% Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ============================================================
# 5. PREPROCESSING & MODEL TRAINING
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
model = AdalineGD(n_iter=100, eta=0.0001)
model.fit(X_train_std, y_train)


# ============================================================
# 6. PREDICTION & EVALUATION
# ============================================================

predictions = model.predict(X_test_std)

print("\n" + 30*"---")
print("CLASSIFICATION REPORT")
print(30*"---")
print(classification_report(y_test, predictions, target_names = None))

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_true=y_test, y_pred=predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=None)

print(30*"---")
print("Output Confusion Matrix:")
print(conf_matrix)
print(30*"---")

disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Wine Classification - All Features")
#plt.title("Confusion Matrix: Wine Classification - Alcohol & Ash")
print(30*"---")
plt.savefig("result/confusion_catrix_wine_classification_adaline_all_features.png")
#plt.savefig("result/confusion_catrix_wine_classification_adaline_alcohol_&_ash.png")
plt.show()


plt.plot(range(1, len(model.cost_) + 1), model.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
#plt.title('Adaline - Learning Rate 0.0001 - Alcohol & Ash')
#plt.savefig("result/adaline_Learning_Rate_0_0001_Alcohol_&_Ash.png")
plt.title('Adaline - Learning Rate 0.0001 - All features')
plt.savefig("result/adaline_learning_rate_0_0001_all_features.png")
plt.show()
