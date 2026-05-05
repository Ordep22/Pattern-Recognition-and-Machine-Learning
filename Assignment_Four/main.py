import os
import math
import numpy as np
import pandas as pd
import scipy.io as wavefile
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Criate a matrix 900/3 splited in to two features

'''
    id   x   y   x
f1  0    
.
.
.
f1 100
f2 101
.
.
.
f2 900


'''
# ==========================================
# 1. GERAÇÃO DO DATASET 1 (Gaussianas)
# ==========================================

# Classe 1: 100 vetores (média zero, matriz de covariância S1)
# Dica: Use a função np.random.multivariate_normal para gerar os dados
id = list(range(0,900,1))

num_samples_S1  = 100
mean  = [0, 0, 0]
S1 = [
    [0.5, 0, 0], 
    [0, 0.5, 0], 
    [0, 0, 0.01]]
samples_s1 = np.random.multivariate_normal(mean, S1, num_samples_S1)


# Classe 2: 800 vetores divididos em 8 grupos de 100
# Dica: Crie cada grupo com sua respectiva média (m1 a m8) e matriz de covariância S2, 
# onde a = 20 e m_i^2 são as médias dadas no PDF.
# Concatene todos os dados em uma única matriz de vetores (900 x 3).
num_samples_S2  = 900
mean  = [0, 0, 0]
S2 = [
    [1, 0, 0], 
    [0, 1, 0], 
    [0, 0, 0.01]]
samples_s2 = np.random.multivariate_normal(mean, S2, num_samples_S2)


m1 = list(np.matrix([20,0,0]).transpose())
m1_sample_s2 = samples_s2[101:200]*m1

print(m1)
print(m1_sample_s2)



# ------------------------------------------
# Passo a) Visualização 3D
# ------------------------------------------
# Crie uma figura 3D e plote os vetores das duas classes usando cores ou marcadores diferentes.
# Adicione um comando para rotacionar ou visualizar de diferentes ângulos.


# ------------------------------------------
# Passo b) LDA (Linear Discriminant Analysis)
# ------------------------------------------
# Instancie o modelo LDA.
# Ajuste o LDA aos dados e projete-os no subespaço bidimensional.
# Plote os dados projetados e comente os resultados obtidos (como as classes se separaram).


# ==========================================
# 2. GERAÇÃO DO DATASET 2 (Espiral 3D)
# ==========================================

# Parâmetros definidos no enunciado
a = 0.1
theta_init = 0.5
theta_fin = 2.05 * np.pi
theta_step = 0.2

# Crie os valores de theta usando a função np.arange
# Para cada valor de theta, calcule r, x e y
# Crie a estrutura z variando de -1 até 1 com passo 0.2 (11 pontos)
# Combine (x, y, z) para formar os pontos da espiral

# ------------------------------------------
# Plotagem da Espiral 3D
# ------------------------------------------
# Plote a espiral tridimensional com as especificações do enunciado:
# - Pontos da mesma espiral 2-dimensional com o mesmo marcador.
# - Grupos de pontos (x, y fixos) com a mesma cor ao longo de z.

# ------------------------------------------
# Passo a) PCA Linear
# ------------------------------------------
# Instancie o PCA com 2 componentes principais.
# Ajuste e transforme o conjunto de dados da espiral.
# Plote o resultado da projeção 2D.

# ------------------------------------------
# Passo b) Kernel PCA
# ------------------------------------------
# Instancie o KernelPCA com dimensão m=2.
# Teste diferentes parâmetros de kernel (como o valor de gamma).
# Plote os resultados e compare com o PCA linear.





