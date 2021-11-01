# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 10:21:50 2021

@author: Albert
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Boston-filtered.csv")
df1 = df
df1.columns = [i for i in range(13)]

def Naive_regression(y):
  C = np.arange(min(y),max(y),0.01)
  SSE = np.zeros(len(C))
  for i in range(len(C)):
    Constant = C[i]*np.ones(len(y))
    SSE[i] = ((Constant-y)**2).sum()
    
  MSE = SSE/len(y)
  MSE = MSE.tolist()
  index = MSE.index(min(MSE))
  
  return C[index], MSE[index]

#c, Linear regression with single attributes
def Single_LR(x,y):
  x = np.array(x)
  y = np.array(y)
  reshape_x = x.reshape((-1,1))
  model = LinearRegression().fit(reshape_x,y)
  slope = model.coef_[0]
  intercept = model.intercept_
  
  MSE = ((slope*x+intercept-y)**2).sum()/len(x)
  return slope, intercept, MSE
    
 #print('slope:',model.coef_)
 #print('intercept:',model.intercept_)
 
#Linear regression using all attributes 
#x= np.array(df[[i for i in range(12)]])
#y = np.array(df[[12]]) 
def All_LR(x,y,dataframe):
  x = np.array(x)
  y = np.array(y)
  model = LinearRegression().fit(x,y)
  slope = model.coef_[0]
  intercept = model.intercept_
  
  y = dataframe[12]#amazing
  
  MSE = (((x*slope).sum(axis=1)+intercept-y)**2).sum()/len(y)
  
  return slope, intercept, MSE
  
def dataset_split(df):
  flag = np.random.randint(0,df.shape[0])
  if flag + df.shape[0]//3 < df.shape[0]:
    test_set = set([i for i in range(flag,flag+df.shape[0]//3)])
    train_set = set([i for i in range(df.shape[0])])-test_set
    test = df.ix[test_set]
    train = df.ix[train_set]
  else:
    test_set = set([i for i in range(flag,df.shape[0])])| \
    set([i for i in range(df.shape[0]//3-(df.shape[0]+1-flag))])
    train_set = set([i for i in range(df.shape[0])])-test_set
    test = df.ix[test_set]
    train = df.ix[train_set]
  
  return train,test

def main_a():
  av_MSE_train = 0
  av_MSE_test = 0
  for i in range(20):
    train,test = dataset_split(df)
    C, MSE_train = Naive_regression(train[12])
    MSE_test = ((C-test[12])**2).sum()/len(test[12])
    av_MSE_train += MSE_train
    av_MSE_test += MSE_test
    
  return av_MSE_train/20,av_MSE_train/20
    
  print("The average MSE of Naive regression on train set is:",av_MSE_train/20)
  print("The average MSE on Naive regression on test set is:",av_MSE_test/20)
  
def main_c():
  space = np.zeros(2*12).reshape(12,2)
  for j in range(12):
    av_MSE_train = 0
    av_MSE_test = 0
    for i in range(20):
      train,test = dataset_split(df)
      slope, intercept, MSE_train = Single_LR(train[j],train[12])
      MSE_test = ((slope*test[j]+intercept-test[12])**2).sum()/len(test[12])
      av_MSE_train += MSE_train
      av_MSE_test += MSE_test
    space[j][0]=av_MSE_train/20
    space[j][1]=av_MSE_test/20
    
  return space
      
def main_d():
  av_MSE_train = 0
  av_MSE_test = 0
  for i in range(20):
    train,test = dataset_split(df)
    x = np.array(train[[i for i in range(12)]])
    y = np.array(train[[12]])
    x_test = np.array(test[[i for i in range(12)]])
    y_test = test[12]
    slope, intercept, MSE_train = All_LR(x,y,train)
    MSE_test = (((x_test*slope).sum(axis=1)+intercept-y_test)**2).sum()/len(y_test)
    av_MSE_train += MSE_train
    av_MSE_test += MSE_test
    
  return av_MSE_train/20,av_MSE_test/20
    
if __name__ == '__main__':
  A = main_a()
  print("The average MSE of Naive regression on train set is:",A[0])
  print("The average MSE of Naive regression on test set is:",A[1])
  
  C=main_c()
  print("The average MSE of single attribute:")
  print(C)
  
  D = main_d()
  print("The average MSE of all attributes regression on train set is:",D[0])
  print("The average MSE of all attributes regression on test set is:",D[1])
  



    
  
    
    