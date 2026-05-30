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
from scipy.cluster.hierarchy import linkage, dendrogram


# ============================================================
# 1. VARIBLE DEFINITIONS & INITIALIZATIONS
# ============================================================

#DEFINES
DEBUG = False
DF = 3

#INITIALIZATIONS
files_path = []
file_sequence = ['df_one','df_two','df_three','df_four']
df_list = []

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

for iten in range(0,4,1):
    df_list.append(pd.read_csv(files_path[iten], header=None, sep = r'\s+',engine='python'))

if DEBUG:

    print("\n" + 30*"---")
    print("DATASET VISUALIZATION")
    print(30*"---") 
    print(df_list[DF].head())
    print("\n")


# ============================================================
# 2.1 INFRASTRUCTURE: DYNAMIC DIRECTORY CREATION
# ============================================================
result_folder = base_path / 'result' / f'result_{file_sequence[DF]}'
result_folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. STANDARLIZATION
# ============================================================

# Initialize the scaler
scaler = StandardScaler()

#Fit and transform the unsupervised data
X_scaled = scaler.fit_transform(df_list[DF])


# ============================================================
# 4. ELBOW LOOP
# ============================================================

wcss = []
silhouette_scores = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels  = kmeans.fit_predict(X_scaled)
    wcss.append(kmeans.inertia_) 

    if k > 1:
        score  = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(score)

# ============================================================s
# 5. RESULT VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title(f'Elbow Method for Data Segmentation - {file_sequence[DF]}')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.savefig(f'{result_folder}' + '/elbow_method_for_knn_trail_data_segmentation'+ f'_{str(file_sequence[DF])}' +'.png')
plt.show()

plt.figure(figsize=(8,6))
plt.plot(range(2,11), silhouette_scores, marker = 's', linestyle = '-', color = 'indigo')
plt.title(f'Silhouette Coeficients vs Number of Clusters - {file_sequence[DF]}')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Silhouette Score')
plt.grid(True)
plt.savefig(f'{result_folder}' + '/silhouette_coeficients_vs_number_of_clusters'+ f'_{str(file_sequence[DF])}' + '.png')
plt.show()

# ============================================================
# 6. HIERARCHICAL CLUSTERING & DENDROGRAM
# ============================================================

plt.figure(figsize = (10,7))
Z  = linkage(X_scaled, method = 'complete')
R = dendrogram(
               Z,
               truncate_mode = 'lastp',
               p = 12, 
               leaf_rotation=90, 
               leaf_font_size=10,
               show_contracted=True
               )
plt.title(f'Hierarchical Clustering Dendrogram (Complete-Link) - {file_sequence[DF]}')
plt.xlabel('Sample Index (or Cluster Size)')
plt.ylabel('Threshold Distance')
plt.grid(False)
plt.savefig(f'{result_folder}' +'/hierarchical_dendrogram_'+ f'{str(file_sequence[DF])}' + '.png')
plt.show()


# ============================================================
# 7. FINAL CLUSTERED DATA VISUALIZATION (THE TARGET PROJECTION)
# ============================================================
"""
STUDY NOTE ON CLUSTER VISUALIZATION:
Once the optimal 'k' is selected via Elbow/Silhouette analysis, we re-train 
the model using this fixed value to inspect the spatial boundary definitions.
Plotting the data points colored by their assigned cluster, along with the 
calculated centroids, provides a visual validation of the algorithm's performance.
"""
best_index = np.argmax(silhouette_scores)
generated_optimal_k = best_index + 2

final_kmeans = KMeans(n_clusters=generated_optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
final_labels = final_kmeans.fit_predict(X_scaled)
centroids = final_kmeans.cluster_centers_

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=final_labels, cmap='viridis', alpha=0.6, edgecolors='k', label='Data Points')
plt.scatter(centroids[:,0], centroids[:,1],marker = '*', color = 'red', s = 200, label = 'Centroids' )
plt.title(f'Final K-Means Segmentation (k = {generated_optimal_k}) - {file_sequence[DF]}')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(f'{result_folder}' +'/final_clusters_distribution_'+ f'{str(file_sequence[DF])}' + '.png')
plt.show()