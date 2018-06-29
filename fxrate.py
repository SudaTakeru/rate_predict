# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 19:59:51 2017

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
from keras.layers import UpSampling1D
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy.random import *

#f = open('nikkeiave.txt', 'r')
f = open('fxrate.txt', 'r')
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

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)-1]

def create_dataset(dataset, look_back=1,timedelay=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        xset = []
        a = dataset[i:(i+look_back)]
        xset.append(a)    
        dataY.append(dataset[i + look_back+timedelay]) 
        dataX.append(xset)
    return numpy.array(dataX), numpy.array(dataY)

timedelay = 1
look_back = 50
trainX, trainY = create_dataset(train, look_back, timedelay)
testX, testY = create_dataset(test, look_back, timedelay)

batch=10
epochs=10
lrc=0.001
day=50
future=750
beta_1c=0.9
daylong=30

opt = keras.optimizers.Adam(lr=lrc, beta_1=beta_1c, beta_2=0.999, epsilon=1e-08, decay=0.0)
def mean_pred(y_true, y_pred):
    return K.mean(y_true-y_pred)

model = Sequential()
model.add(LSTM(batch, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['accuracy', mean_pred])

## training
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch + 1, epochs))
    itmax = int(trainX.shape[0] / batch)
 
    for i in range(itmax):
        hist=model.train_on_batch(trainX, trainY)

## 短期予測
trainPredict = model.predict(trainX)
testPredict_shortL=[[]for j in range(len(testX)-1)];
testPredict_short=[]
for i in range(len(testX)-1):    
    testPredict_shortL[i] = model.predict(testX[i:i+1])
    testPredict_short.append(testPredict_shortL[i][0][0])

## 長期予測
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

testPredictL=[[]for j in range(future)];
for ii in range(future):
    testPredictLb = model.predict(testX[0+ii:ii+look_back])
    testPredictX2 = predict_dataset(testPredictLb)
    testPredictX = testPredictX2[:,:,:,0]
    
    testPredict2=list(testPredictLb)
    for i in range(day):
        testPredict3 = []
        testPredictLb = model.predict(testPredictX)
        testPredict2.extend(list(testPredictLb))
        testPredict3.extend(testPredict2[len(testPredict2)-look_back:len(testPredict2)])
        testPredict3.extend(testPredictLb)
        #    hist=model.train_on_batch(testPredictX, testPredict+0.0001*(-1+2*rand()))
        hist=model.train_on_batch(testPredictX, testPredictLb)
        testPredictX2 = predict_dataset(testPredict3)
        testPredictX = testPredictX2[:,:,:,0] 
    testPredictL[ii]=testPredict2

#pad_col = numpy.zeros(dataset.shape)
#def pad_array(val):
#    return numpy.array([numpy.insert(pad_col, 0, x) for x in val])

plt.plot(trainPredict,label='predict')
plt.plot(trainY,label='true')
plt.legend()
plt.title('train')
plt.show()

plt.plot(testPredict_short,label='predict')
plt.plot(testY,label='true')
plt.legend()
plt.title('short term predict')
plt.savefig("short_{}_{}.png".format(lrc, beta_1c))
plt.show()


#
#trainc=numpy.power(trainPredict-trainY,2)
#testc=numpy.power(testPredict-testY,2)
#trainmse=trainc.sum/len(trainY)
#testmse=testc.sum/len(testY)

## trainingの評価

error=[[]for j in range(future)];

for i in range(len(testPredictL)):
    plt.plot(range(i,i+day),testPredictL[i][:day])
    error2=[]
    for ii in range(day):
        RMSV=math.sqrt((testPredictL[i][ii][0]-numpy.transpose(testY[i+ii]))**2)
        error2.append(RMSV)        
    error[i]=error2

## n日後予測毎errorの平均
erroreach=list(range(day))
errorave=list(range(day))
for ii in range(day):
    erroreach2=[]
    for i in range(len(testPredictL)):
        erroreach2.append(error[i][ii])
    erroreach[ii]=erroreach2
    errorave[ii]=numpy.sum(erroreach[ii])/len(testPredictL)
         
plt.plot(testY,label='true')
plt.legend()
plt.title('long term predict')
plt.savefig("{}_long_{}_{}.png".format(day,lrc, beta_1c))
plt.show()

plt.plot(errorave)
plt.title('error average for day')
plt.show()

##信頼区間
Value=1.96*errorave[day-1]


## up or down accuracy
day2=day

true=list(range(day))
truerate=list(range(day))
true2=list(range(day2))
truerate2=list(range(day2))
for i in range(len(testPredictL)):        
    for ii in range(day):
        if (testPredictL[i][ii][0]-testPredictL[i][ii+1][0])>0:
            if (numpy.transpose(testY[i+ii])-numpy.transpose(testY[i+ii+1]))>0:
                true[ii]=true[ii]+1
        else:
            if (numpy.transpose(testY[i+ii])-numpy.transpose(testY[i+ii+1]))<0:
                true[ii]=true[ii]+1
                
for ii in range(day):
    truerate[ii]=true[ii]/len(testPredictL)

for i in range(len(testPredictL)):
    for d in range(1,day2):
        if (testPredictL[i][0][0]-testPredictL[i][0+d][0])>0:
            if (testY[i+0]-testY[i+0+d])>0:
                true2[d]=true2[d]+1
        else:
            if (testY[i+0]-testY[i+0+d])<0:
                true2[d]=true2[d]+1
for d in range(1,day2):            
    truerate2[d]=true2[d]/len(testPredictL)

plt.plot(truerate)
plt.plot(truerate2[1:],label='from one day')
plt.title('truerate for day')
plt.show()

## 今日から先を知りたい時用

testPredictLb = model.predict(testX[len(testX)-look_back:])
testPredictX2 = predict_dataset(testPredictLb)
testPredictX = testPredictX2[:,:,:,0]
testPredict2long=list(range(2))
testPredict2long[0]=testX[len(testX)-1][0][19]
testPredict2long[1]=testPredictLb[look_back-1]
for i in range(daylong-1):
     testPredict3 = []
     testPredictLb = model.predict(testPredictX)
     testPredict2long.extend(list(testPredictLb))
     testPredict3.extend(testPredictX[0][0][1:look_back])
     testPredict3.extend(testPredictLb)
     #hist=model.train_on_batch(testPredictX, testPredictLb)
     testPredictX2 = predict_dataset(testPredict3)
     testPredictX = testPredictX2
    
testPredict2long2=list(range(len(testPredict2long)))
testPredict2long3=list(range(len(testPredict2long)))
for i in range(len(testPredict2long)):
     testPredict2long2[i]=testPredict2long[i]+Value
     testPredict2long3[i]=testPredict2long[i]-Value                 

plt.plot(range(len(testX),len(testX)+daylong+1),testPredict2long)
plt.plot(range(len(testX),len(testX)+daylong+1),testPredict2long2,label='upper')
plt.plot(range(len(testX),len(testX)+daylong+1),testPredict2long3,label='lower')
plt.plot(testY,label='true')
plt.legend()
plt.title('long term predict')
plt.savefig("{}_longafter.png".format(daylong))
plt.show()



