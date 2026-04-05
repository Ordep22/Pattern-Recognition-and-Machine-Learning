import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
#from ucimlrepo import fetch_ucirepo, list_available_datasets
import sklearn as skl
from sklearn.datasets import load_wine
import sys




class AdalineGD:

    def __init__(self,eta = 0.01, n_iter = 50, randon_state = 1):
        
        self.eta = eta
        self.n_iter  = n_iter
        self.randon_state = randon_state
        self.data = None
        self.df = None
        self.X = None
        self.Y = None

    def load_Data(self):
        try:
            self.data  = load_wine()
            self.df = pd.DataFrame(data = self.data['data'], columns=self.data['feature_names'])
            self.df['target'] = self.data['target']
            self.df['class'] = self.df['target'].map(lambda ind: self.data['target_names'][ind])
            print(self.df.head)

            self.X = self.df.iloc[0:100, [0,2]].values
            self.Y = self.df.iloc[0:100,4].values
            #Y = np.were(Y == "SELECT the TIPE")

            plt.scatter(self.X[:50, 0], self.X[:50, 1],
            color='red', marker='o', label='class')
            plt.scatter(self.X[50:100, 0], self.X[50:100, 1],
            color='blue', marker='s', label='alcohol')

            plt.xlabel("Class")
            plt.ylabel("Alcohol")
            plt.legend(loc='upper left')

            plt.show()

        except:
            print("Something went wrong! Check the libriry imports")
            sys.exit()

    def fit(self,X,Y):
        rgen = np.random.RandomState(self.randon_state)
        self.w_ =rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        self.b_ = np.float(0.)
        self.losses_ = []
    
        for i in range(self.n_iter):
            net_input  = self.net_input(X)
            output  = self.activation(net_input)
            erros = (Y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(erros)/X.Shape[0]
            self.b_ += self.eta *2.0 * erros.mean()
            loss = (erros**2).mean()
            self.losses_.append(loss)

        return self

    def net_input(self, X):
        return np.dot(X,self.w_) + self.b_

    def activation(self,X):
        return X
    
    def predic(self, X):
        return np.where(self.activation(self.net_input(X) >= 0.5,1,0))



    def show_analyses(self):

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

        ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
        ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker='o')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('log(Mean squared error)')
        ax[0].set_title('Adaline - Learning rate 0.1')
        
        #ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
        #ax[1].plot(range(1, len(ada2.losses_) + 1), ada2.losses_, marker='o')
        #ax[1].set_xlabel('Epochs')
        #ax[1].set_ylabel('Mean squared error')
        #ax[1].set_title('Adaline - Learning rate 0.0001')
        
        # plt.savefig('images/02_11.png', dpi=300)
        plt.show()


def main():

    ada1 = AdalineGD(n_iter=15, eta=0.1)
    ada1.load_Data()
    ada1.fit(ada1.X,ada1.Y)
    ada1.show_analyses()




if __name__ == '__main__':
    main()