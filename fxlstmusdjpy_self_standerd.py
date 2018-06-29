# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 13:00:40 2018

@author: buryu-
"""


import numpy
import matplotlib.pyplot as plt
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
from statistics import mean, median,variance,stdev


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
#dataset = scaler.fit_transform(dataset)
data=kabuka
data.extend(recentdata)
datamm = scaler.fit_transform(data)
dataset=datamm

train_size = int(len(dataset) * 0.5)
test_size = len(dataset)*(1-0.2) - train_size
demo_size = len(dataset)-train_size-test_size

m = mean(data[:train_size])
v = variance(data[:train_size])
dataset=numpy.array(data)
dataset=(dataset-m)/v
dataset=list(dataset)

ti=train_size+test_size
train= dataset[0:train_size]
test= dataset[train_size:int(ti)]
demo =  dataset[int(ti):len(dataset)]

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
demoX, demoY = create_dataset(demo, look_back, timedelay)

batch=1
epochs=10
lrc=0.001
day=50
future=test_size-look_back-timedelay
beta_1c=0.9
daylong=50

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

model.save_weights('fxparam.hdf5')
json_string = model.to_json()
open('fxmodel.json', 'w').write(json_string)
model.save('my_model.h5')

## 短期予測
trainPredict = model.predict(trainX)
testPredict_shortL=[[]for j in range(len(testX)-1)];
testPredict_short=[]
for i in range(len(testX)-1):    
    testPredict_shortL[i] = model.predict(testX[i:i+1])
    testPredict_short.append(testPredict_shortL[i][0][0])


lpdata= dataset[train_size-look_back:int(ti)]
lpX, lpY = create_dataset(lpdata, look_back, timedelay)

## 長期予測
def predict_dataset(dataset):
    setdata=[]
    #for i in range(len(dataset)):
    setx=[]
    setx.append(dataset[0:look_back])
    setdata.append(setx)
    return numpy.array(setdata)

testPredictL=[[]for j in range(int(future-look_back))]
for ii in range(int(future- look_back)):
    testPredictLb = model.predict(lpX[ii:ii+look_back])
    testPredictX2 = predict_dataset(testPredictLb[-look_back:])
    testPredictX = testPredictX2[:,:,:,0]
    testPredictLb0 = testPredictLb
    testPredict2=list(testPredictLb[-1])
    for i in range(day):
        testPredict3 = []
        testPredictLb = model.predict(testPredictX)
        testPredict2.extend(testPredictLb[0])
        testPredict3.extend(testPredictLb0[len(testPredictLb0)-look_back:len(testPredictLb0)])
        testPredict3.extend(testPredictLb)
        numpy.append(testPredictLb0,testPredictLb)
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

#
#trainc=numpy.power(trainPredict-trainY,2)
#testc=numpy.power(testPredict-testY,2)
#trainmse=trainc.sum/len(trainY)
#testmse=testc.sum/len(testY)

## trainingの評価

error=[[]for j in range(int(future))];
plt.figure(figsize=(20,20),dpi=200)
for i in range(len(testPredictL)):
    plt.plot(range(i,i+day),testPredictL[i][:day])
    error2=[]
    for ii in range(day):
        RMSV=math.sqrt((testPredictL[i][ii]*v-numpy.transpose(testY[i+ii]*v))**2)
        error2.append(RMSV)        
    error[i]=error2


## n日後予測毎errorの平均
erroreach=list(range(day))
errorave=list(range(day))
standarderror=list(range(day))
for ii in range(day):
    erroreach2=[]
    for i in range(len(testPredictL)):
        erroreach2.append(error[i][ii])
    erroreach[ii]=erroreach2
    errorave[ii]=numpy.sum(erroreach[ii])/len(testPredictL)
    standarderror[ii]=errorave[ii]/math.sqrt(len(testPredictL))
    
    
plt.plot(testY,label='true')
plt.legend()
plt.title('long term predict')
plt.savefig("long.png")
plt.show()

plt.plot(errorave)
plt.title('RMSV for day')
plt.show()

plt.plot(standarderror)
plt.title('standarderror for day')
plt.show()
##信頼区間
Values=list(range(day))
for i in range(day):
    Values[i]=1.96*errorave[i]

## 1日後
Value=1.96*errorave[0]
testPredict_short2=list(range(len(testPredict_short)))
testPredict_short3=list(range(len(testPredict_short)))
for i in range(len(testPredict_short)):
     testPredict_short2[i]=testPredict_short[i]+Value
     testPredict_short3[i]=testPredict_short[i]-Value
                      
plt.plot(testPredict_short,label='predict')
plt.plot(testY,label='true')
plt.plot(testPredict_short2,label='upper')
plt.plot(testPredict_short3,label='lower')
plt.legend()
plt.title('short term predict')
plt.savefig("short_{}_{}.png".format(lrc, beta_1c))
plt.show()





## up or down accuracy
day2=day

true=list(range(day))
truerate=list(range(day))
true2=list(range(day2))
truerate2=list(range(day2))
for i in range(len(testPredictL)):        
    for ii in range(day):
        if (testPredictL[i][ii]-testPredictL[i][ii+1])>0:
            if (numpy.transpose(testY[i+ii])-numpy.transpose(testY[i+ii+1]))>0:
                true[ii]=true[ii]+1
        else:
            if (numpy.transpose(testY[i+ii])-numpy.transpose(testY[i+ii+1]))<0:
                true[ii]=true[ii]+1
                
for ii in range(day):
    truerate[ii]=true[ii]/len(testPredictL)

for i in range(len(testPredictL)):
    for d in range(1,day2):
        if (testPredictL[i][0]-testPredictL[i][0+d])>0:
            if (testY[i+0]-testY[i+0+d])>0:
                true2[d]=true2[d]+1
        else:
            if (testY[i+0]-testY[i+0+d])<0:
                true2[d]=true2[d]+1
for d in range(1,day2):            
    truerate2[d]=true2[d]/len(testPredictL)

plt.plot(truerate)
plt.plot(truerate2[1:])
plt.title('truerate for day')
plt.show()

