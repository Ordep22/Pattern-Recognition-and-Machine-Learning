"""
Assignment 05: Clustering

Course: Pattern Recognition and Machine Learning - UTFPR

Objective: Determine the optimal number of clusters using 
various clustering approaches (e.g., K-Means and Hierarchical Clustering)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score


# ============================================================
# 1. VARIBLE DEFINITIONS & INITIALIZATIONS
# ============================================================

#DEFINES
DEBUG = False

#INITIALIZATIONS
files_path = []

# ============================================================
# 2. DATA LOADING & PATH MANAGEMENT
# ============================================================

# STUDY NOTE: 
# Using Path(__file__) ensures the script finds the data folder 
# regardless of where the terminal is running from.

base_path  = Path(__file__).resolve().parent
data_dir = base_path / 'data'

# Locating the .txt file
file = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
file  = sorted(file)



for i in file: files_path.append(data_dir / i)

# STUDY NOTE: header=None is mandatory for raw UCI datasets to avoid 
# treating the first sample as column names.
df_four =  pd.read_csv(files_path[3], header=None, sep = r'\s+',engine='python')
df_three = pd.read_csv(files_path[2], header=None, sep = r'\s+',engine='python')
df_two =   pd.read_csv(files_path[1], header=None, sep = r'\s+',engine='python')
df_one =   pd.read_csv(files_path[0], header=None, sep = r'\s+',engine='python')

if DEBUG:

    print("\n" + 30*"---")
    print("DATASET VISUALIZATION")
    print(30*"---") 
    print(df_one.head())
    print("\n")

    print("\n" + 30*"---")
    print("DATASET VISUALIZATION")
    print(30*"---") 
    print(df_two.head())
    print("\n")

    print("\n" + 30*"---")
    print("DATASET VISUALIZATION")
    print(30*"---") 
    print(df_three.head())
    print("\n")

    print("\n" + 30*"---")
    print("DATASET VISUALIZATION")
    print(30*"---") 
    print(df_four.head())
    print("\n")


# ============================================================
# 3. STANDARLIZATION
# ============================================================

# Initialize the scaler
scaler = StandardScaler()

#Fit and transform the unsupervised data
X_scaled = scaler.fit_transform(df_one)


# ============================================================
# 4. ELBOW LOOP
# ============================================================

# Loop through different numbers of clusters (k = 1 to 10)
wcss = []
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_) # inertia_ stores the Within-Cluster Sum of Squares
    silhouette_scores.append(kmeans.labels_)

# ============================================================
# 5. RESULT VISUALIZATION
# ============================================================

# Plot the Elbow Curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Kananaskis Trail/Data Segmentation')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.show()