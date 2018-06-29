# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 22:35:34 2018

@author: buryu-
"""
import numpy
import matplotlib.pyplot as plt
import pandas
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers import LSTM
from keras.layers import UpSampling1D
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy.random import *
from keras.models import load_model

batch=1
epochs=100
lrc=0.001
day=50
future=750
beta_1c=0.9
daylong=50
timedelay = 1
look_back = 50

opt = keras.optimizers.Adam(lr=lrc, beta_1=beta_1c, beta_2=0.999, epsilon=1e-08, decay=0.0)
def mean_pred(y_true, y_pred):
    return K.mean(y_true-y_pred)

model = Sequential()
model.add(LSTM(batch, input_shape=(1, look_back)))
model.add(Dense(1))

model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['accuracy', mean_pred])
# model reconstruction from JSON:
#from keras.models import model_from_json
#model = load_model('my_model.h5')
#json_string=open('fxmodel.json', 'r')
#model = model_from_json(json_string)
model.load_weights('fxparam.hdf5')

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

f2 = open('fxrate213.txt', 'r')
line=f2.readlines()
linesize=len(line)
recentdata=[]
for i in range(0,linesize):
  recentdata.append(float(line[i]))
f2.close()

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

trainX, trainY = create_dataset(train, look_back, timedelay)
testX, testY = create_dataset(test, look_back, timedelay)

ii=250
plt.plot(testY[ii:ii+day],label='true')

plt.plot(range(day),testPredictL[ii][:day],label='predict')
plt.title('long term predict')
plt.legend()
plt.show()
goalday=day

for i in range(ii,ii+goalday):
    plt.plot(range(i-ii,goalday),testPredictL[i][:goalday-(i-ii)])
    
#plt.show()

plt.plot(testY[ii:ii+day],label='true')
plt.legend()
plt.title('until one day')
plt.show()


data=kabuka
data.extend(recentdata)
datamm = scaler.fit_transform(data)
dataX, dataY = create_dataset(datamm, look_back, timedelay)

def predict_dataset(dataset):
    setdata=[]
    #for i in range(len(dataset)):
    setx=[]
    setx.append(dataset[0:look_back])
    setdata.append(setx)
    return numpy.array(setdata)

'''
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
    
testPredict2long2=list(range(len(testPredict2long)-1))
testPredict2long3=list(range(len(testPredict2long)-1))
for i in range(len(testPredict2long)-1):
     testPredict2long2[i]=testPredict2long[i]+Values[i]
     testPredict2long3[i]=testPredict2long[i]-Values[i]               

plt.plot(range(len(testX),len(testX)+daylong+1),testPredict2long)
plt.plot(range(len(testX),len(testX)+daylong),testPredict2long2,label='upper')
plt.plot(range(len(testX),len(testX)+daylong),testPredict2long3,label='lower')
plt.plot(range(len(testX),len(testX)+daylong),datamm[test_size+train_size:test_size+train_size+daylong],label='true')
plt.plot(testY,label='true')
plt.legend()
plt.title('long term predict')
plt.show()
'''

#plt.plot(range(len(datamm[train_size:train_size+test_size]),datamm[train_size:train_size+test_size])
#plt.plot(range(timedelay+look_back,len(testY)+timedelay+look_back),testY)
