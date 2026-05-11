import matplotlib.pyplot as plt
import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap


#Debug == True to activete the debug informations
DEBUG  = True

iris  = datasets.load_iris()
X = iris.data[:,[2,3]]
Y = iris.target


if DEBUG:
    print("DEBUG - ON\n") 
    print(f"Class labels: {np.unique(Y)}\n")


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, 
                                                    random_state = 1, stratify = Y )


if DEBUG:
    print(10*"---")
    print("DEBUG - ON\n") 
    print(f"Labels count in Y: {np.bincount(Y)}")
    print(f"Labels count in Y_test: {np.bincount(Y_test)}")
    print(f"Labels count in Y_train: {np.bincount(Y_train)}\n")
    print(10*"---")




sc =StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn  = Perceptron(max_iter = 40, eta0 = 0.1, random_state = 1)
ppn.fit(X_train_std,Y_train)

Y_pred = ppn.predict(X_test_std)

if DEBUG:
    print(10*"---")
    print("DEBUG - ON\n")
    print(f"Size of Y Test: {len(Y_test)}") 
    #print(f"Misclassified sample: {((Y_test != Y_pred).sum())/len(Y_test)*100:0.2} %\n")
    print(f"Accurace: {ppn.score(X_test_std, Y_test):0.2}")
    print(10*"---")



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
    plt.xlim(XX1.min(), XX1.max())
    plt.xlim(XX2.min(), XX2.max())

    for idx, cl in enumerate(np.unique(Y)):
        plt.scatter(x = X[Y == cl, 0], y = X[Y == cl, 0], alpha= 0.8, 
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
        


X_combined_std = np.vstack((X_train_std, X_test_std))
Y_combined = np.hstack((Y_train, Y_test))
plot_decision_regios(X=X_combined_std,Y = Y_combined, 
                     classfier=ppn, test_idx=range(105,150))
plt.xlabel(f'Petal Length [standardized]')
plt.ylabel(f'Petal width [standardized]')
plt.legend(loc = 'upper left')
plt.show()

