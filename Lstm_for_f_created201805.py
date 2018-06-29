# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:50:50 2018

@author: buryu-
"""
## 1日後の予測

import numpy as np
import matplotlib.pyplot as plt
import math
import keras
from keras.layers import Dense, Activation, Reshape
from keras.layers import LSTM,Input
from keras.models import Model
from keras.layers import UpSampling1D
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy.random import *

#lerning parameter
day=1
lookback = 50
dim=1
epochs=50
mergin=5

# Adam parameter
lrc=0.001
beta_1c=0.9


f = open('fxrate.txt', 'r')
line=f.readlines()
linesize=len(line)
kabuka=[]
for i in range(0,linesize):
  kabuka.append(float(line[i]))
f.close()
dataset=kabuka

f2 = open('fxrate213.txt', 'r')
line=f2.readlines()
linesize=len(line)
recentdata=[]
for i in range(0,linesize):
  recentdata.append(float(line[i]))
f2.close()

data=kabuka
data.extend(recentdata)
plt.plot(data)
plt.title('training')
plt.show()
data=np.array([data])

#正規化
datamax=np.max(data)+mergin
datamin=np.min(data)-mergin
data=(data-datamin)/(datamax-datamin)

plt.title('standard')     
plt.plot(list(data)[0])
plt.show()


def creatdataset(da,look_back):
    infdim=da.shape[0]
    Tlength=da.shape[1]
    dataset=np.zeros((infdim,look_back,Tlength-look_back))
    for i in range(Tlength-look_back):
        dataset[:,:,i] = da[:,i:i+look_back]
        
    return dataset

# Create data set
trainind=round(data.shape[1]*(2/3))
train=creatdataset(data[:,:trainind],lookback)
test=creatdataset(data[:,trainind:-day],lookback)

target=np.array(data[:,lookback+day:trainind+day])
testtarget=np.array(data[:,lookback+trainind+day:])

opt = keras.optimizers.Adam(lr=lrc, beta_1=beta_1c, beta_2=0.999, epsilon=1e-08, decay=0.0)
def mean_pred(y_true, y_pred):
    return K.mean(y_true-y_pred)

# Netework
units=1
inputs = Input(shape=(dim,lookback))
predictions = LSTM(units, activation='relu')(inputs)
model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['accuracy', mean_pred])

hist=list(range(epochs))
trainloss=list(range(epochs))
for epoch in range(epochs):
    hist[epoch]=model.fit(np.transpose(train,axes=(2,0,1)),np.transpose(target,axes=(1,0)),verbose=0)
    trainloss[epoch]=hist[epoch].history['loss']

plt.title('trainloss')
plt.plot(trainloss)
plt.show()
k=1
predict=list(range(test.shape[2]-k))
predictn=list(range(test.shape[2]-k))
for i in range(test.shape[2]-k):
    #p=model.predict(np.array([np.transpose(test,axes=(2,0,1))[i]]),batch_size=1)
    p=model.predict(np.transpose(test,axes=(2,0,1))[i:k+i,:,:],batch_size=k)
    predict[i]=p[0][0]
    predictn[i]=p[0][0]*(datamax-datamin)+datamin

plt.title('test_standard')
plt.plot(list(testtarget)[0],label='true')
plt.plot(predict,label='predict')
plt.legend()
plt.show()

plt.title('test')
plt.plot(list(testtarget)[0]*(datamax-datamin)+datamin,label='true')
plt.plot(predictn,label='predict')
plt.legend()
plt.show()


