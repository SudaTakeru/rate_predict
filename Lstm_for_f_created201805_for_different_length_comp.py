# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:18:10 2018

@author: buryu-
"""

## maxday日後までの予測

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
import random
from statistics import mean, median,variance,stdev


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
    
maxday=5

underthresholderrorper=list(range(maxday-1))
meanerror=list(range(maxday-1))
goodper=list(range(maxday-1))


for t in range(1,maxday):
    
    #lerning parameter
    day=1#day日分の予測２以上だとうまくいかない・・・
    dayafter=t#dayafter日後を予測
    lookback = 50
    dim=1
    epochs=10
    
    threshold=0.5
    
    # Adam parameter
    lrc=0.001
    beta_1c=0.9
    
    dataori=kabuka
    dataori.extend(recentdata)
    #plt.plot(data)
    #plt.title('training')
    #plt.show()
    data=np.array([dataori])
    trainind=round(data.shape[1]*(2/3))
    #正規化
#    datamax=np.max(data)+mergin
#    datamin=np.min(data)-mergin
#    data=(data-datamin)/(datamax-datamin)
    #標準化
    m = mean(dataori[:trainind])
    v = variance(dataori[:trainind])
    data=(data-m)/v
    #plt.title('standard')     
    #plt.plot(list(data)[0])
    #plt.show()
    
    # Create data set
    
    train=creatdataset(data[:,:trainind],lookback)
    test=creatdataset(data[:,trainind+dayafter:-dayafter],lookback)
    
    trainrandind=random.randrange(train.shape[2])
    
    target=creattargetdataset(data[:,lookback:trainind+dayafter],day,train.shape[2])
    testtarget=creattargetdataset(data[:,lookback+trainind+dayafter:],day,test.shape[2])
    
    #
    if t==1:
        plt.title('test')
        plt.plot(list(testtarget[0,0,:]*v+m),label='true')
    
    opt = keras.optimizers.Adam(lr=lrc, beta_1=beta_1c, beta_2=0.999, epsilon=1e-08, decay=0.0)
    def mean_pred(y_true, y_pred):
        return K.mean(y_true-y_pred)
    
    # Netework
    units=day
    inputs = Input(shape=(dim,lookback))
    predictions = LSTM(units, activation='relu',kernel_initializer='he_uniform')(inputs)
    model = Model(inputs=inputs, outputs=predictions)
    
    model.compile(loss='mean_squared_error',
          optimizer=opt,
          metrics=['accuracy', mean_pred])
    
    # Training
    hist=list(range(epochs))
    trainloss=list(range(epochs))
    for epoch in range(epochs):
        hist[epoch]=model.fit(np.transpose(train,axes=(2,0,1)),np.transpose(target,axes=(2,0,1))[:,0,:],verbose=0)
    
    # Testing
    k=1
    predict=list(range(test.shape[2]))
    predictn=list(range(test.shape[2]))
    good=0
    for i in range(test.shape[2]):
    #p=model.predict(np.array([np.transpose(test,axes=(2,0,1))[i]]),batch_size=1)
        p=model.predict(np.transpose(test,axes=(2,0,1))[i:k+i,:,:],batch_size=k)
        predict[i]=p[0]
        predictn[i]=p[0]*v+m
        if (test[0,lookback-1,i]-testtarget[0,0,i])*(test[0,lookback-1,i]-predictn[i][0])>0:
            good=good+1
    
    goodper[t-1]=good/test.shape[2]
    error=np.sqrt(np.square(testtarget[0][0]*v+m-np.array(predictn)[:,0]))
    
    meanerror[t-1]=sum(error)/len(error)
    count=0
    for i in range(len(error)):
        if error[i]<threshold:
            count=count+1
    underthresholderrorper[t-1]=count/len(error)
    #print(meanerror)
    #print(underthresholderrorper)
    num=str(t)
    plt.plot(list(np.array(predictn)[:,0]),label='predict_'+num)


plt.legend()
plt.savefig("predicts.png")
plt.show()    
    
plt.plot(range(1,maxday),meanerror)
plt.title('meanerror')
plt.savefig("meanerror.png")
plt.show()

plt.plot(range(1,maxday),underthresholderrorper)
plt.title('underthresholderrorper')
plt.savefig("underthresholderrorper.png")
plt.show()

plt.plot(range(1,maxday),goodper)
plt.title('true percentage up or down')
plt.savefig("true_percentage_up_or_down.png")
plt.show()

