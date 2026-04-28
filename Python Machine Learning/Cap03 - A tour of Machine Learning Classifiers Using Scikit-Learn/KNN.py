import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap

def plot_decision_regios(X,Y,classfier, test_idx = None, resolution = 0.02):
    #Setup Marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors  = ('red','blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(Y))])

    #plot the decision surface
    X1_min, X1_max = X[:, 0].min() - 1, X[:,0].max() +1
    X2_min, X2_max = X[:, 1].min() - 1, X[:,1].max() +1

    XX1, XX2 = np.meshgrid(np.arange(X1_min, X1_max, resolution), 
                           np.arange(X2_min, X2_max, resolution))
    Z = classfier.predict(np.array([XX1.ravel(),XX2.ravel()]).T)
    Z = Z.reshape(XX1.shape)
    plt.contourf(XX1,XX2, Z, alpha = 0.3, cmap = cmap)
    
    #plt.xlim(XX2.min(), XX2.max())

    for idx, cl in enumerate(np.unique(Y)):
        plt.scatter(x = X[Y == cl, 0], y = X[Y == cl, 1], alpha= 0.8, 
                    marker = markers[idx],
                    c = colors[idx], label = cl)
        

    #Highlight test sample
    if test_idx:
        #Plot all sample
        X_test, Y_test = X[test_idx, :], Y[test_idx]

        plt.scatter(X_test[:,0],X_test[:,1],
                    c =  colors[0], edgecolors= 'black', alpha= 1.0,
                    linewidths=1, marker= 'o',
                    s = 100, label  = 'Test Set')

    plt.xlim(XX1.min(), XX1.max())

    return Z


iris = datasets.load_iris()
x = iris.data[:,[2,3]]
y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 1, stratify = y)

knn  = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
knn.fit(X_train, Y_train)

test_predictions = knn.predict(X_test)

X_combined = np.vstack((X_train, X_test))
Y_combined = np.hstack((Y_train, Y_test))
plot_decision_regios(X= X_combined,Y = Y_combined, classfier = knn, test_idx = range(105,150))
plt.title('Randon Forest')
plt.xlabel(f'Petal Length [cm]')
plt.ylabel(f'Petal width [cm]')
plt.legend(loc = 'upper left')
plt.show()

# Model Evaluation: Generating predictions and computing 
print(30*"---"+"\n")
print("Classification Report - TEST SET (Setosa, Versicolor & Virginica):")
print(classification_report(Y_test, test_predictions, target_names=['Setosa', 'Versicolor', 'Virginica']))
print(30*"---"+"\n")
