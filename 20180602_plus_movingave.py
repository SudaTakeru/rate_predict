# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 20:01:35 2018

@author: buryu-
"""

import numpy as np
import fi 
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
import random
from statistics import mean, median,variance,stdev

# parmeter
timedelay = 1
look_back = 50
dim=4
epochs=10

# Evaluate parameter
future=50
threshold=0.5
trainper=2/3

# Adam parameter
lrc=0.001
beta_1c=0.9

fun=fi.fi(look_back,timedelay,dim)
    
## data読み込み
f = open('fxrate.txt', 'r')
line=f.readlines()
linesize=len(line)
kabuka=[]
for i in range(0,linesize):
  kabuka.append(float(line[i]))
f.close()

f2 = open('fxrate213.txt', 'r')
line=f2.readlines()
linesize=len(line)
recentdata=[]
for i in range(0,linesize):
  recentdata.append(float(line[i]))
f2.close()
    
## data整理
dataori=kabuka
dataori.extend(recentdata)
#plt.plot(data)
#plt.title('training')
#plt.show()
dataset=dataori

train_size = int(len(dataori) * trainper)

#標準化
m = mean(dataori[:train_size])
v = variance(dataori[:train_size])
dataset=np.array(dataori)
dataset=(dataset-m)/v
dataset=list(dataset)

#moving average
types = [5,10,20] 
maxmoving=max(types)
mA=list(range(len(types)+1))
trainmA=list(range(len(types)+1))
testmA=list(range(len(types)+1))
for i in range(len(types)):
    move=types[i]
    mA[i]=list(range(len(dataset)-maxmoving))
    for ii in range(len(dataset)-maxmoving):
        mA[i][ii]=np.sum(dataset[ii:ii+move])/move
    trainmA[i]=mA[i][:train_size-maxmoving]
    testmA[i]=mA[i][train_size-maxmoving:]

trainmA[len(types)]= dataset[maxmoving:train_size]
testmA[len(types)]= dataset[train_size:]
trainX, trainY = fun.create_dataset(trainmA)
testX, testY = fun.create_dataset(testmA)

##モデル
opt = keras.optimizers.Adam(lr=lrc, beta_1=beta_1c, beta_2=0.999, epsilon=1e-08, decay=0.0)
def mean_pred(y_true, y_pred):
    return K.mean(y_true-y_pred)
    
model=fun.nn()
model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['accuracy', mean_pred])
    
# Training
hist=list(range(epochs))
trainloss=list(range(epochs))
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch + 1, epochs))
    #hist[epoch]=model.fit(trainX,trainY,verbose=0)
    itmax = int(trainX.shape[0] / 1)
    for i in range(itmax):
        hist=model.train_on_batch(trainX, trainY)


## 短期予測 
testPredict_shortL=[[]for j in range(len(testX)-1)];
testPredict_short=[]

truerate=list(range(1,15))
goodrate=list(range(1,15))
undererrorrate=list(range(1,15))

for ii in range(1,15):
    threshold=ii*0.1
    true=0
    count=0
    good2=0
    undererror=0
    for i in range(len(testX)-1):   
        good=0
        if ii==1:
            testPredict_shortL[i] = model.predict(testX[i:i+1])
            testPredict_short.append(testPredict_shortL[i][0][dim-1])
        if 0<(testY[i][dim-1]-testX[i][dim-1][-1])*(testPredict_shortL[i][0][dim-1]-testX[i][dim-1][-1]):
            true=true+1
            good=1
        if math.sqrt(((testY[i][dim-1]-testX[i][dim-1][-1])*v)**2)>threshold:
            count=count+1
            if good:
                good2=good2+1
        if threshold>math.sqrt((math.sqrt(((testY[i][dim-1]-testX[i][dim-1][-1])*v)**2)-math.sqrt(((testPredict_shortL[i][0][dim-1]-testX[i][dim-1][-1])*v)**2))**2):
            undererror=undererror+1
            
    truerate[ii-1]=true/(len(testX)-1)
    goodrate[ii-1]=good2/count
    undererrorrate[ii-1]=undererror/(len(testX)-1)

plt.plot(testY*v+m,label='true')
plt.plot(np.array(testPredict_short)*v+m)
plt.title('short term predict')
plt.legend()
plt.show()

x=0.1*np.array(range(1,15))

plt.plot(x,truerate,label='truerate')
plt.plot(x,goodrate,label='goodrate')
plt.plot(x,undererrorrate,label='undererror')
plt.ylim([0,1])
plt.legend()
plt.show


