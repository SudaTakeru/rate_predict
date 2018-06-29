# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 22:02:36 2017

@author: buryu-
"""


import numpy
import matplotlib.pyplot as plt
import pandas
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy.random import *

f = open('nikkeiave.txt', 'r')
#data = f.read()      #readですべて読み込む
line=f.readlines()
#for row in reader2:
#    print row
linesize=len(line)
kabuka=[]
for i in range(0,linesize):
  kabuka.append(float(line[i]))
f.close()
dataset=kabuka

Minka=min(kabuka)
Maxka=max(kabuka)
kabukacov=numpy.cov(kabuka)
kabukamean=sum(kabuka)/len(kabuka)

scaler = MinMaxScaler(feature_range=(Minka, Maxka))
dataset = scaler.fit_transform(dataset)
#dataset = (dataset-kabukamean)/kabukacov #標準化
dataset=(dataset-Minka)/(Maxka-Minka) #正規化

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)-1]

def create_dataset(dataset, look_back=1,timedelay=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-timedelay):
        xset = []
        a = dataset[i:(i+look_back)]
        xset.append(a)    
        dataY.append(dataset[i + look_back+timedelay]) 
        dataX.append(xset)
    return numpy.array(dataX), numpy.array(dataY)



def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

testPredict_short=[]
testPredict_long=[]
for j in range(30):
    print('day {}/{}'.format(j+ 1,30))
    timedelay = j
    look_back = j+5
    trainX, trainY = create_dataset(train, look_back, timedelay)
    testX, testY = create_dataset(test, look_back, timedelay)

    batch=5
    epochs=100
    lrc=0.001
    beta_1c=0.90

    opt = keras.optimizers.Adam(lr=lrc, beta_1=beta_1c, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model = Sequential()
    model.add(LSTM(batch, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['accuracy', mean_pred])

    #hist=model.fit(trainX, trainY, epochs=300, batch_size=5, verbose=1)
    for epoch in range(epochs):
        #print('Epoch {}/{}'.format(epoch + 1, epochs))
        itmax = int(trainX.shape[0] / batch)
        for i in range(itmax):
            hist=model.train_on_batch(trainX, trainY)

    trainPredict = model.predict(trainX)
    testPredict_short2 = model.predict(trainX[-look_back:])
    testPredict_short.append(testPredict_short2)

    def predict_dataset(dataset):
        setdata=[]
        #for i in range(len(dataset)):
        setx=[]
        setx.append(dataset[0:look_back])
        setdata.append(setx)
        return numpy.array(setdata)

    testPredict = model.predict(testX[0:look_back])
    testPredictX2 = predict_dataset(testPredict)
    testPredictX = testPredictX2[:,:,:,0]

    testPredict2=list(testPredict)
    for i in range(len(testX)):
        testPredict3 = []
        testPredict = model.predict(testPredictX)
        testPredict2.extend(list(testPredict))
        testPredict3.extend(testPredict2[len(testPredict2)-look_back:len(testPredict2)])
        testPredict3.extend(testPredict)
        #    hist=model.train_on_batch(testPredictX, testPredict+0.0001*(-1+2*rand()))
        hist=model.train_on_batch(testPredictX, testPredict)
        testPredictX2 = predict_dataset(testPredict3)
        testPredictX = testPredictX2[:,:,:,0] 
    
    testPredict_long.append(testPredict2)

#g2=g*kabukacov+kabukamean
testY2=testY*(Maxka-Minka)+Minka
          
testPredict_shortz=[]
plt.figure(figsize=(8,6))
for i in range(len(testPredict_short)):
    testPredict_short3=testPredict_short[i]*(Maxka-Minka)+Minka 
   # testPredict_shorts=[testPredict_short3[-1],i] 
    testPredict_shorts=testPredict_short3[-1]    
    testPredict_shortz.append(testPredict_shorts)
    
plt.plot(testPredict_shortz,label='predict')

plt.plot(testY2,label='true')
plt.legend()
plt.title('short term predict')
plt.savefig("short_to_long.png")
plt.show()
plt.close()

plt.figure(figsize=(8,6))
for i in range(len(testPredict_long)):
    a=testPredict_long[i]
    c=[]
    for j in range(len(a)):
        b=a[j][0]
        c.append(b)
    d=numpy.array(c)
    testPredict_long3=d*(Maxka-Minka)+Minka
    plt.plot(tuple(testPredict_long3),label='predict')                                

plt.plot(testY2,label='true')
plt.legend()
plt.title('long term predict')
plt.savefig("long.png")
plt.show()
#plt.close()

