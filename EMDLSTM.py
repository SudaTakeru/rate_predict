# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 00:58:35 2017

@author: buryu-
"""
from PyEMD import EMD
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

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

emd = EMD()
IMFs = emd(dataset)
#
#train_size = int(len(dataset) * 0.67)
#test_size = len(dataset) - train_size
#train, test = dataset[0:train_size], dataset[train_size:len(dataset)-1]
#
#def create_dataset(dataset, look_back=1,timedelay=1):
#    dataX, dataY = [], []
#    for i in range(len(dataset)-look_back-1):
#        xset = []
#        a = dataset[i:(i+look_back)]
#        xset.append(a)    
#        dataY.append(dataset[i + look_back+timedelay]) 
#        dataX.append(xset)
#    return numpy.array(dataX), numpy.array(dataY)
#
#timedelay = 1
#look_back = 12
#trainX, trainY = create_dataset(train, look_back, timedelay)
#testX, testY = create_dataset(test, look_back, timedelay)
#
#batch=5
#epochs=100
#lrc=0.001
#day=10
#beta_1c=0.9
#
#opt = keras.optimizers.Adam(lr=lrc, beta_1=beta_1c, beta_2=0.999, epsilon=1e-08, decay=0.0)
#def mean_pred(y_true, y_pred):
#    return K.mean(y_pred)
#
#model = Sequential()
#model.add(LSTM(batch, input_shape=(1, look_back)))
#model.add(Dense(1))
#model.compile(loss='mean_squared_error',
#              optimizer=opt,
#              metrics=['accuracy', mean_pred])
#
##hist=model.fit(trainX, trainY, epochs=300, batch_size=5, verbose=1)
#for epoch in range(epochs):
#    print('Epoch {}/{}'.format(epoch + 1, epochs))
#    itmax = int(trainX.shape[0] / batch)
# 
#    for i in range(itmax):
#        hist=model.train_on_batch(trainX, trainY)
#
#trainPredict = model.predict(trainX)
#testPredict_shortL=[[]for j in range(len(testX)-1)];
#testPredict_short=[]
#for i in range(len(testX)-1):    
#    testPredict_shortL[i] = model.predict(testX[i:i+1])
#    testPredict_short.append(testPredict_shortL[i][0][0])
#
#
#def predict_dataset(dataset):
#    setdata=[]
#    #for i in range(len(dataset)):
#    setx=[]
#    setx.append(dataset[0:look_back])
#    setdata.append(setx)
#    return numpy.array(setdata)
#
#testPredict = model.predict(testX[0:look_back])
#testPredictX2 = predict_dataset(testPredict)
#testPredictX = testPredictX2[:,:,:,0]
#
#testPredictL=[[]for j in range(350)];
#for ii in range(350):
#    testPredictLb = model.predict(testX[0+ii:ii+look_back])
#    testPredictX2 = predict_dataset(testPredictLb)
#    testPredictX = testPredictX2[:,:,:,0]
#    
#    testPredict2=list(testPredictLb)
#    for i in range(day):
#        testPredict3 = []
#        testPredictLb = model.predict(testPredictX)
#        testPredict2.extend(list(testPredictLb))
#        testPredict3.extend(testPredict2[len(testPredict2)-look_back:len(testPredict2)])
#        testPredict3.extend(testPredictLb)
#        #    hist=model.train_on_batch(testPredictX, testPredict+0.0001*(-1+2*rand()))
#        hist=model.train_on_batch(testPredictX, testPredictLb)
#        testPredictX2 = predict_dataset(testPredict3)
#        testPredictX = testPredictX2[:,:,:,0] 
#    testPredictL[ii]=testPredict2
#
##pad_col = numpy.zeros(dataset.shape)
##def pad_array(val):
##    return numpy.array([numpy.insert(pad_col, 0, x) for x in val])
#
#plt.plot(trainPredict,label='predict')
#plt.plot(trainY,label='true')
#plt.legend()
#plt.title('train')
#plt.show()
#
#plt.plot(testPredict_short,label='predict')
#plt.plot(testY,label='true')
#plt.legend()
#plt.title('short term predict')
#plt.savefig("short_{}_{}.png".format(lrc, beta_1c))
#plt.show()
#
##
##trainc=numpy.power(trainPredict-trainY,2)
##testc=numpy.power(testPredict-testY,2)
##trainmse=trainc.sum/len(trainY)
##testmse=testc.sum/len(testY)
#for i in range(len(testPredictL)):
#    plt.plot(range(i,i+day),testPredictL[i][:day])
#plt.plot(testY,label='true')
#plt.legend()
#plt.title('long term predict')
#plt.savefig("{}_long_{}_{}.png".format(day,lrc, beta_1c))
#plt.show()
##
##loss = hist.history['loss']
## 
### lossのグラフ
##plt.plot(range(epochs), loss, marker='.', label='loss')
##plt.legend(loc='best', fontsize=10)
##plt.grid()
##plt.xlabel('epoch')
##plt.ylabel('loss')
##plt.show()
##
##acc = hist.history['acc']
##
##plt.plot(range(epochs), acc, marker='.', label='acc')
##plt.legend(loc='best', fontsize=10)
##plt.grid()
##plt.xlabel('epoch')
##plt.ylabel('acc')
##plt.show()

