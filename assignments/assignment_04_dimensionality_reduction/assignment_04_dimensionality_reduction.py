"""
Assignment 04: Dimensionality Reduction

Course: Pattern Recognition and Machine Learning - UTFPR

Objective: Apply fundamental dimensionality reduction techniques to synthetic datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, KernelPCA
# ============================================================
# 1. DATASET GENERATION
# ============================================================

# --- Class 1 Generation ---
# 100 vectors centered around mean zero with covariance matrix S1
num_samples_S1  = 100
mean_s1  = [0, 0, 0]

#Cavariance matrix
S1 = [
    [0.5, 0, 0], 
    [0, 0.5, 0], 
    [0, 0, 0.01]
    ]

samples_s1 = np.random.multivariate_normal(mean_s1, S1, num_samples_S1)


print("\n" + 30*"---")
print("DATASET VISUALIZATION - CLASS 1")
print(30*"---")
print(f"Shape of Class 1: {samples_s1.shape}")

# --- Class 2 Generation ---
# 800 vectors total, divided into 8 clusters of 100 samples each.
# Each cluster is shifted to its respective mean vector (m1 to m8).


"""
STUDY NOTE ON TRANSPOSE MATRIX:
In the mathematical literature of Linear Algebra and Pattern Recognition, 
by universal convention, every pattern vector is considered a column vector.

m1 = [20, 0, 0]T = | 20 |
                   | 0  |
                   | 0  | 

In NumPy, using a 1D array like np.array([20, 0, 0]) allows us to take advantage 
of 'Broadcasting'. The array is automatically applied across the specified slice 
without needing explicit matrix transposition.
"""

num_samples_S2  = 800
mean_s2  = [0, 0, 0]

#Cavariance matrix
S2 = [
    [1, 0, 0], 
    [0, 1, 0], 
    [0, 0, 0.01]]

samples_s2 = np.random.multivariate_normal(mean_s2, S2, num_samples_S2)

#Matrix one
m1 = np.array([20,0,0])
samples_s2[0:100]+= m1


#Matrix two 
m2 = np.array([10,10,0])
samples_s2[100:200] += m2

#Matrix three
m3 = np.array([0,20,0]) 
samples_s2[200:300] += m3


#Matrix four
m4 = np.array([-10,10,0]) 
samples_s2[300:400] += m4


#Matrix five
m5 = np.array([-20,0,0]) 
samples_s2[400:500] += m5

#Matrix six
m6 = np.array([-10,-10,0]) 
samples_s2[500:600]+= m6


#Matrix seven
m7 = np.array([0,-20,0]) 
samples_s2[600:700] += m7


#Matrix Eight
m8 = np.array([10,-10,0]) 
samples_s2[700:800]+= m8


print("\n" + 30*"---")
print("DATASET VISUALIZATION - CLASS 2")
print(30*"---")
print(f"Shape of Class 2: {samples_s2.shape}")

# --- Dataset Concatenation ---
# Merging Class 1 and Class 2 into a single matrix (900 x 3)

samples  = np.concatenate((samples_s1,samples_s2),axis=0)
print(type(samples))


print("\n" + 30*"---")
print("DATASET VISUALIZATION")
print(30*"---")
print(samples)



# --- Creatre a DataFrame from an array ---
class_label  = np.concatenate((np.zeros(100), np.ones(800)), axis = 0)

sample_dataset = pd.DataFrame(
    {'Class_label':class_label, 
     'X_axis': samples[:,0],                           
     'Y_axis': samples[:,1], 
     'Z_axis':samples[:,2] 
    })


# ============================================================
# 2. 3D VISUALIZATION
# ============================================================

"""
STUDY NOTE ON VISUALIZATION:
Our dataset is composed of distinct data samples across two classes in a 3D space. 
Therefore, a 3D Scatter Plot is required rather than a surface plot. 
This allows us to visually inspect spatial distribution and class separability.
"""

#Plot the points in to the 3D space

fig  = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

#Classe One - 0 to 100
ax.scatter3D(samples[:100,0], samples[:100,1], samples[:100, 2], color = 'blue', marker = 'o',
              label = 'Classe 1 (Core)')

#Classe two - 101 to 900
ax.scatter3D(samples[101:,0], samples[101:,1], samples[101:,2], color = 'gray', marker = '^', 
             label  = 'Classe 2 (Surrounding Rings)')


ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Scatter Plot: Spatial Class Distribution')
ax.legend()

# Set initial viewing angle for optimal perspective
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.savefig("result/tridimensional_scatter_plot_spatial_class_distribution.png")
plt.show()

# ============================================================
# 3. LINEAR DISCRIMINANT ANALYSIS (LDA)
# ============================================================


# Isolate features (X) and targets (y) using standard Pandas column selection
X = sample_dataset[['X_axis', 'Y_axis', 'Z_axis']]
y = sample_dataset['Class_label']

# Applying LDA to reduce the dimensionality
# For a 2-class problem, max components = C - 1 = 1 dimension.
lda = LinearDiscriminantAnalysis(n_components = 1)
X_lda = lda.fit_transform(X,y)

# Plotting the 1D Projected Result
plt.figure(figsize=(8, 6))

# Creating a flat dummy array for the Y-axis to plot 1D output onto a linear baseline
y_dummy = np.zeros(len(X_lda))

# Plotting samples belonging to Class 0
plt.scatter(X_lda[y == 0, 0], y_dummy[y == 0], 
            color='blue', alpha=0.7, label='Class 1 (Core)', marker='o')

# Plotting samples belonging to Class 1
plt.scatter(X_lda[y == 1, 0], y_dummy[y == 1], 
            color='gray', alpha=0.5, label='Class 2 (Rings)', marker='^')

plt.title('LDA 1D Projection: Class Separability Matrix')
plt.xlabel('Linear Component 1')
plt.ylabel('Dummy Axis (Zero Alignment)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='best')

plt.tight_layout()
plt.savefig("result/lda_onedimensional_projection.png")
plt.show()


# ============================================================
# 2. DATASET GENERATION 2 (3D Spiral Extrusion)
# ============================================================

a = 0.1
theta_init = 0.5
theta_fin = 2.05 * np.pi
theta_step = 0.2

theta = np.arange(theta_init, theta_fin, theta_step)
z_space = np.arange(-1, 1.1, 0.2)  

r = a * theta
x = r * np.cos(theta)
y = r * np.sin(theta)

"""
STUDY NOTE ON 3D SPIRAL EXTRUSION:
The 3D spiral dataset is not a continuous helix (spring shape). Instead, it represents 
a 2D Archimedean spiral extruded along the Z-axis. 
Every fixed (x, y) coordinate pair forms a vertical linear group spanning across all Z values.
To fulfill the requirements, points on the same 2D plane share the same marker, 
and vertical groups (fixed x, y) share the same color.
"""

print("\n" + 30*"---")
print("DATASET VISUALIZATION - SPIRAL GENERATION")
print(30*"---")
print(f"Number of spiral points (theta): {len(theta)}")
print(f"Number of vertical levels (z): {len(z_space)}")

# ------------------------------------------------------------
# 3D Spiral Plotting
# ------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.viridis(np.linspace(0, 1, len(theta)))
spiral_point = []

for i in range(len(theta)):

    x_line = np.full_like(z_space, x[i])
    y_line = np.full_like(z_space, y[i])
    ax.scatter3D(x_line, y_line, z_space, color=colors[i], marker='o', s=30)    
    ax.plot3D(x_line, y_line, z_space, color=colors[i], linestyle=':', alpha=0.4)

    for x_val, y_val, z_val in zip(x_line, y_line, z_space):
        spiral_point.append([x_val, y_val, z_val])


X_spiral = np.array(spiral_point)


ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Extrusion Plot: Archimedean Spiral Sheet')

ax.view_init(elev=20, azim=60)

plt.tight_layout()
plt.savefig("result/tridimensional_spiral_extrusion_plot.png")
plt.show()

# ============================================================
# 3. LINEAR PRINCIPAL COMPONENT ANALYSIS (PCA)
# ============================================================

"""
STUDY NOTE ON LINEAR PCA:
Linear PCA identifies the orthogonal axes (principal components) that maximize 
the variance of the data. For this extruded 3D spiral, the first two components 
are expected to capture the planar structure of the Archimedean spiral, 
effectively flattening the vertical Z-axis variance.
"""

pca  = PCA(n_components = 2)

X_pca = pca.fit_transform(X_spiral)

print("\n" + 30*"---")
print("LINEAR PCA TRANSFORMATION")
print(30*"---")
print(f"Original 3D shape: {X_spiral.shape}")
print(f"Reduced 2D PCA shape: {X_pca.shape}")

# ------------------------------------------------------------
# Plotting the 2D PCA Projection
# ------------------------------------------------------------
plt.figure(figsize= (8,6))
plt.scatter(X_pca[:,0], X_pca[:,1],color = 'teal', alpha = 0.6, marker = 'o')

plt.title('Linear PCA: 2D Projection of 3D Spiral')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("result/linear_pca_2d_projection.png")
plt.show()

# ============================================================
# 4. KERNEL PRINCIPAL COMPONENT ANALYSIS (KERNEL PCA)
# ============================================================

"""
STUDY NOTE ON KERNEL PCA:
Unlike Linear PCA, Kernel PCA maps the non-linear structure of the 3D spiral 
into a higher-dimensional feature space using the 'Kernel Trick'. 
By applying a Gaussian (RBF) kernel, the model attempts to unfold the non-linear 
manifold. The parameter 'gamma' dictates the radius of influence of the kernel:
- Low gamma values behave similarly to Linear PCA.
- High gamma values may cause overfitting, scattering the geometric structure.
- An optimal gamma will successfully linearize or cluster the spiral pattern.
"""

gamma_values = [0.1, 1, 5, 10]

for g in gamma_values:
    
    kpca = KernelPCA(kernel ='rbf', gamma = g, n_components = 2)
    
    X_kpca = kpca.fit_transform(X_spiral)
    
    print("\n" + 30*"---")
    print(f"KERNEL PCA TRANSFORMATION - GAMMA: {g}")
    print(30*"---")
    print(f"Reduced 2D Kernel PCA shape: {X_kpca.shape}")
    
    # --------------------------------------------------------
    # Plotting the 2D Kernel PCA Projection
    # --------------------------------------------------------
    
    plt.figure(figsize=(8, 6))
    

    plt.scatter(X_kpca[:,0], X_kpca[:,1],color = 'teal', alpha = 0.6, marker = 'o')
    
    
    plt.title(f'Kernel PCA: 2D Projection (RBF Kernel, Gamma = {g})')
    plt.xlabel('Kernel Component 1')
    plt.ylabel('Kernel Component 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    plt.savefig(f"result/kernel_pca_gamma_{g}.png")
    plt.show()






