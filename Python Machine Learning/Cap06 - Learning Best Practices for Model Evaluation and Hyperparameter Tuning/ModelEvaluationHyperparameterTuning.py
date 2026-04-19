import pandas as pd
import numpy as np
from matplotlib import pylab as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline



'''
1 - We start by reading the dataset directly from the UCI website usinf Pandas
'''
try:

    df  = pd.read_csv('/Users/pedropereira/Documents/Work/' \
    'Pattern-Recognition-and-Machine-Learning/Python Machine ' \
    'Learning/Cap06 - Learning Best Practices for Model Evaluation and Hyperparameter ' \
    'Tuning/breast+cancer+wisconsin+diagnostic/wdbc.data', header= None)
    print(30*"---")
    print("File successfully readed")
    print(30*"---")
    #print(df.head())
    #print(30*"---")


except:

    print("Was not possible red the file 'wdbc.data'")


'''
2 - Next, we assign thw 30 feature to a Numpy array x. Using tha Pandas methods, 
we transform the class label from their original string representation ('M' and 'B') into intergers:

'''


X = df.loc[:, 2:].values
Y = df.loc[:, 1 ].values
le = LabelEncoder()
Y = le.fit_transform(Y)
df[1] = df[1].map({'M':1, 'B':0})

print(30*"---")
print(df.head())
print(30*"---")

'''
3 - Before we construct our first model pipeline in the following subsections let us
dividethe dataset in to a separated training dataset (80 percent of the data)
and a separated test dataser (20 percent of the data)
'''

X_train,  X_test, Y_train, Y_test = train_test_split(X ,Y, test_size = 0.20, stratify = Y, random_state = 1)

#print(30*"---")
#print("Undertanting the out put of 'train_test_split'\n")
#print("X_test\n")
#print(f"Type: {type(X_train)}")
#print(f"{X_train}")
#print(30*"---")

pipe_lr = make_pipeline(StandardScaler(),PCA(n_components = 2), LogisticRegression(random_state = 1))
pipe_lr.fit(X_train, Y_train)
Y_pred = pipe_lr.predict(X_test)
print(f'Test Accuracy:{pipe_lr.score(X_test, Y_test):.03f}')









