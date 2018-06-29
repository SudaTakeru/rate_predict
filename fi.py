# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 11:10:23 2018

@author: buryu-
"""
import numpy
import keras
from keras.layers import LSTM,Input,Dense
from keras.models import Model

class fi:
    def __init__(self,lookback,timedelay,dim):
        self.look_back = lookback
        self.time_delay = timedelay
        self.dim= dim
        
    def create_dataset(self, dataset):
        if self.dim==1:
            dataX, dataY = [], []
            for i in range(len(dataset)-self.look_back-self.time_delay):
                xset = []
                a = dataset[i:(i+self.look_back)]
                xset.append(a)    
                dataY.append(dataset[i + self.look_back+self.time_delay]) 
                dataX.append(xset)
            return numpy.array(dataX), numpy.array(dataY)
        else:
            dataX, dataY = [], []
            for i in range(len(dataset[0])-self.look_back-self.time_delay):
                xset = []
                a=list(range(self.dim))
                b=list(range(self.dim))
                for ii in range(self.dim):
                    a[ii] = dataset[ii][i:(i+self.look_back)]
                    b[ii] = dataset[ii][i + self.look_back+self.time_delay]
                xset.append(a)    
                dataY.append(b) 
                dataX.append(xset)
            return numpy.array(dataX)[:,0,:,:], numpy.array(dataY)

    def nn(self):
        units=self.dim
        inputs = Input(shape=(self.dim,self.look_back))
        x = LSTM(units)(inputs)
        predictions = Dense(units)(x)
        model = Model(inputs=inputs, outputs=predictions)
        return model

    def long_predict_data_modify(self,data1,data2):
        #data1.shape=[1,dim,lookback]
        #data2.shape=[dim,1]
        newdata=data1[0:1,:,1:]
        newdata=numpy.append(newdata,numpy.array([data2]),axis=2)
        return newdata
    
