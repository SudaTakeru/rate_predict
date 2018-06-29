# -*- coding: utf-8 -*-
"""
Created on Tue May 22 17:47:36 2018

@author: buryu-
"""

## day日後までの予測

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
day=1#day日分の予測２以上だとうまくいかない・・・
dayafter=10#dayafter日後を予測
lookback = 50
dim=1
epochs=100
mergin=5

threshold=0.5

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
#plt.plot(data)
#plt.title('training')
#plt.show()
data=np.array([data])

#正規化
datamax=np.max(data)+mergin
datamin=np.min(data)-mergin
data=(data-datamin)/(datamax-datamin)

#plt.title('standard')     
#plt.plot(list(data)[0])
#plt.show()


def creatdataset(da,look_back):
    infdim=da.shape[0]
    Tlength=da.shape[1]
    dataset=np.zeros((infdim,look_back,Tlength-look_back))
    for i in range(Tlength-look_back):
        dataset[:,:,i] = da[:,i:i+look_back]
        
    return dataset

def creattargetdataset(da,day,num):
    infdim=da.shape[0]
    dataset=np.zeros((infdim,day,num))
    for i in range(num):
        dataset[:,:,i] = da[:,i:i+day]
        
    return dataset

# Create data set
trainind=round(data.shape[1]*(2/3))
train=creatdataset(data[:,:trainind],lookback)
test=creatdataset(data[:,trainind+dayafter:-dayafter],lookback)

target=creattargetdataset(data[:,lookback:trainind+dayafter],day,train.shape[2])
testtarget=creattargetdataset(data[:,lookback+trainind+dayafter:],day,test.shape[2])

opt = keras.optimizers.Adam(lr=lrc, beta_1=beta_1c, beta_2=0.999, epsilon=1e-08, decay=0.0)
def mean_pred(y_true, y_pred):
    return K.mean(y_true-y_pred)

# Netework
units=day
inputs = Input(shape=(dim,lookback))
predictions = LSTM(units, activation='relu')(inputs)
model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['accuracy', mean_pred])

# Training
hist=list(range(epochs))
trainloss=list(range(epochs))
for epoch in range(epochs):
    hist[epoch]=model.fit(np.transpose(train,axes=(2,0,1)),np.transpose(target,axes=(2,0,1))[:,0,:],verbose=0)
    trainloss[epoch]=hist[epoch].history['loss']

plt.title('trainloss')
plt.plot(trainloss)
plt.show()

# Testing
k=1
predict=list(range(test.shape[2]))
predictn=list(range(test.shape[2]))
for i in range(test.shape[2]):
    #p=model.predict(np.array([np.transpose(test,axes=(2,0,1))[i]]),batch_size=1)
    p=model.predict(np.transpose(test,axes=(2,0,1))[i:k+i,:,:],batch_size=k)
    predict[i]=p[0]
    predictn[i]=p[0]*(datamax-datamin)+datamin
            
error=np.sqrt(np.square(testtarget[0][0]*(datamax-datamin)+datamin-np.array(predictn)[:,0]))
meanerror=sum(error)/len(error)
count=0
for i in range(len(error)):
    if error[i]<threshold:
        count=count+1
underthresholderrorper=count/len(error)
print(meanerror)
print(underthresholderrorper)

plt.title('test_standard')
plt.plot(list(testtarget[0][0]),label='true')
#plt.plot(list(data[:,lookback+trainind+day:][0]))
plt.plot(predict,label='predict')
plt.legend()
plt.show()

plt.title('test')
plt.plot(list(testtarget[0][0]*(datamax-datamin)+datamin),label='true')
#plt.plot(list(data[:,lookback+trainind+day:][0]*(datamax-datamin)+datamin),label='true')
plt.plot(predictn,label='predict')
plt.legend()
plt.show()
