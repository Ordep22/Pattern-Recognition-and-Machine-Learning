import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('/Users/pedropereira/Documents/Work/Pattern-Recognition-and-Machine-Learning/Python Machine Learning/Cap10 - Pedricting Continuous Target Variables with Regression Analysis/bottle.csv')
#df_binary = df[['Salnty','T_degC']]
#df_binary.columns = ['Sal', 'Temp']
df.head()

