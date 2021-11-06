# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 21:09:31 2021

@author: Albert
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("Boston-filtered.csv")
df.columns = [i for i in range(13)]

#define \gamma and \sigma
gamma = np.zeros(15)
sigma = np.zeros(13)
for i in range(15):
  gamma[i] = math.pow(2,i-40)
  
for i in range(13):
  sigma[i]=math.pow(2,7+0.5*i)
  
#split train_set and test_set
flag = df.shape[0]//3*2
train_set = [i for i in range(flag)]
test_set = [i for i in range(flag,df.shape[0])]
train = df.ix[train_set]
test = df.ix[test_set]

train = np.array(train)
test = np.array(test)
#calculate kernel K, input:L*n matrix, output: L*L matrix
def K(X,sigma):
  l = X.shape[0]
  K = np.zeros(l*l).reshape(l,l)
  for i in range(l):
    for j in range(l):
      t = np.sum((X[i]-X[j])**2)
      K[i][j] = np.exp(-t/(2*math.pow(sigma,2)))
      
  return K
      
#mse,input: predict_value, real_value(l*1 metrix);output: value
def mse(predict,real):
  return np.sum((predict-real)**2)/predict.shape[0]

#calculate \alpha*
def alpha(K,gamma,y):
  l = len(y)
  return np.linalg.pinv(K+gamma*l*np.eye(l)).dot(y)

#calculate y_predict;
#y_predict--y_test;x--x_test
def y_predict(alpha,sigma,train,x):
  L = len(alpha)
  K = np.zeros(L)
  for i in range(L):
    t = np.sum((train[i]-x)**2)
    K[i] = np.exp(-t/(2*math.pow(sigma,2)))
  y_predict = alpha.dot(K)
  
  return y_predict

#split train set X into k parts 
def kfold(X,k):
  flag = X.shape[0]//k
  set = []
  d = 0
  for i in range(k):
    set.append([j for j in range(d,(i+1)*flag)])
    d = d+flag
    
  t = 0
  for t in range(k):
    yield X[set[t]],X[[j for j in range(X.shape[0]) if j not in set[t]]]
    
def main_a():
  #split train_set and test_set
  train_set_x = train[:,0:12]
  train_set_y = train[:,12]
  test_set_x = test[:,0:12]
  test_set_y = test[:,12]
  k = 5
  #put mse in the array
  MSE = np.zeros(len(gamma)*len(sigma)).reshape(len(gamma),len(sigma))
  for i in range(len(gamma)):
    for j in range(len(sigma)):
      a = kfold(train,k)
      mse_test = 0
      for t in range(k):
        k_test, k_train = next(a)
        K_ = K(k_train[:,0:12],sigma[j])
        alpha_ = alpha(K_,gamma[i],k_train[:,12])
        y_predict_ = np.zeros(k_test.shape[0])
        for l in range(k_test.shape[0]):
          y_predict_[l] = y_predict(alpha_,sigma[j],k_train[:,0:12],k_test[:,0:12][l])
        mse_test += mse(y_predict_,k_test[:,12])
      MSE[i][j]=mse_test/k
      
  index = np.where(MSE == np.min(MSE))
  best_gamma = gamma[index[0][0]]
  best_sigma = sigma[index[1][0]]
  K_ = K(train_set_x,best_sigma)
  alpha_ = alpha(K_,best_gamma,train_set_y)
  y_predict_train = np.zeros(train_set_x.shape[0])
  for i in range(train_set_x.shape[0]):
    y_predict_train[i] = y_predict(alpha_,best_sigma,train_set_x,train_set_x[i])
  mse_train = mse(y_predict_train,train_set_y)
  y_predict_test = np.zeros(test_set_x.shape[0])
  for i in range(test_set_x.shape[0]):
    y_predict_test[i] = y_predict(alpha_,best_sigma,train_set_x,test_set_x[i])
  mse_test = mse(y_predict_test,test_set_y)
  return MSE,mse_train,mse_test
  #choose the minma of mse
'''
  index = np.where(MSE == np.min(MSE))
  best_gamma = gamma[index[0][0]]
  best_sigma = sigma[index[1][0]]
  
  K_ = K(train_set_x,best_sigma)
  alpha_ = alpha(K_,best_gamma,train_set)
  y_predict_train = np.zeros(train_set_x.shape[0])
  for i in range(train_set_x.shape[0]):
    y_predict_train[i] = y_predict(alpha_,best_sigma,train_set_x,train_set_x[i])
  mse_train = mse(y_predict_train,train_set_y)
  y_predict_test = np.zeros(test_set_x.shape[0])
  for i in range(test_set_x.shape[0]):
    y_predict_test[i] = y_predict(alpha_,best_sigma,train_set_x,test_set_x[i])
  #y_predict_test = y_predict(alpha_,best_sigma,train_set_x,test_set_x)
  mse_test = mse(y_predict_test,test_set_y)
  
  return best_gamma,best_sigma,mse_train,mse_test
'''

def main_b():
  MSE = pd.read_csv("mse.csv")
  MSE = np.array(MSE)[:,1:]
  ls = []
  for i in range(MSE.shape[0]):
    for j in range(MSE.shape[1]):
      ls.append([i,j,MSE[i][j]])
  
  ls = np.array(ls)
  x,y,z = ls[:,0]-40,ls[:,1]*0.5+6.5,ls[:,2]
  
  fig = plt.figure()
  ax = fig.add_subplot(111,projection='3d')
  ax.scatter(x,y,z)
  ax.set_xlabel('\gamma')
  ax.set_ylabel('\sigma')
  ax.set_zlabel('mse')
  plt.show()

#input: dataframe; output: dataframe  
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

def main_d():
  mse_ls = np.zeros(20)
  best_gamma = np.zeros(20)
  best_sigma = np.zeros(20)
  for t in range(20):
    train,test = dataset_split(df)
    train_arr = np.array(train)
    test_arr = np.array(test)
    #put mse in the array
    MSE = np.zeros(len(gamma)*len(sigma)).reshape(len(gamma),len(sigma))
    for i in range(len(gamma)):
      for j in range(len(sigma)):
        K_ = K(train_arr[:,0:12],sigma[j])
        alpha_ = alpha(K_,gamma[i],train_arr[:,12])
        y_predict_ = np.zeros(test_arr.shape[0])
        for l in range(test_arr.shape[0]):
          y_predict_[l] = y_predict(alpha_,sigma[j],train_arr[:,0:12],test_arr[:,0:12])
        MSE[i][j] = mse(y_predict_,test_arr[:,12])
    index = np.where(MSE == np.min(MSE))
    best_gamma[t] = gamma[index[0][0]]
    best_sigma[t] = sigma[index[1][0]]
    mse_ls[t] = MSE[index[0][0]][index[1][0]]
    
  return mse_ls,best_gamma,best_sigma
 
if __name__ == '__main__':
  a = main_a()
  main_b()
  d = main_d()
          
    