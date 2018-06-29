# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:59:14 2017

@author: buryu-
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv3D, Conv2D, BatchNormalization
from keras.layers import Activation, Flatten, Dropout, UpSampling2D, MaxPooling2D, Reshape
from sklearn.preprocessing import MinMaxScaler

# nikkei data
f = open('nikkeiave.txt', 'r')
line=f.readlines()
linesize=len(line)
kabuka=[]
for i in range(0,linesize):
  kabuka.append(float(line[i]))
f.close()
dataset=kabuka

Minka=min(kabuka)
Maxka=max(kabuka)
kabukacov=np.cov(kabuka)
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
    for i in range(len(dataset)-look_back-1):
        xset = []
        a = dataset[i:(i+look_back)]
        xset.append(a)    
        dataY.append(dataset[i + look_back+timedelay]) 
        dataX.append(xset)
    return np.array(dataX), np.array(dataY)

timedelay = 1
look_back = 400
trainX, trainY = create_dataset(train, look_back, timedelay)
testX, testY = create_dataset(test, look_back, timedelay)
trainXX=trainX[:,0,:]

# define model
def generator_model():
    model = Sequential()
 
    model.add(Dense(1024, input_shape = (100, )))
    model.add(Activation('tanh'))

    model.add(Dense(1024)) 
    
    #model.add(Reshape((1*12*12*12,), input_shape = (1, 12, 12, 12)))
#    model.add(Conv2D(12, (3, 3), padding = 'same'))
    model.add(Activation('tanh'))
    model.add(Dense(look_back))    
    return model
 
def discriminator_model():
    model = Sequential()
    model.add(Dense(look_back, input_shape = (look_back, )))
#    model.add(Conv2D(12, (5, 5),strides=(2,2), padding='same', input_shape=(12,)))
#    model.add(Activation('tanh'))
#    #model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Conv2D(128, (5, 5), padding='same'))
#    model.add(Activation('tanh'))
#    #model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dense(1024))
    model.add(Activation('tanh'))
 
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
 
 
def combined_model(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model
 
generator = generator_model()
#generator.summary()
 
discriminator = discriminator_model()
#discriminator.summary()
 
discriminator.trainable = False
combined = combined_model(generator, discriminator)
#combined.summary()
 
opt = keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
 
discriminator.trainable = True
discriminator.compile(loss='binary_crossentropy', optimizer=opt)
 
discriminator.trainable = False
combined.compile(loss='binary_crossentropy', optimizer=opt)
 
epochs = 200
batch_size = 10
 
param_folder = './param'
 
if not os.path.isdir(param_folder): 
    os.makedirs(param_folder)
 
 
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch + 1, epochs))
 
    itmax = int(trainX.shape[0] / batch_size)
    progbar = keras.utils.generic_utils.Progbar(target = itmax)
 
    for i in range(itmax):
 
        # train discriminator
        x = trainXX[i * batch_size : (i + 1) * batch_size]
        n = np.array([np.random.uniform(-1, 1,  100) for _ in range(batch_size)])
        g = generator.predict(n, verbose=1)
        y = [1] * batch_size + [0] * batch_size
 
        d_loss = discriminator.train_on_batch(np.concatenate((x, g)), y)
 
        # train generator
        n = np.array([np.random.uniform(-1, 1,  100) for _ in range(batch_size)])
        y = [1] * batch_size
 
        g_loss = combined.train_on_batch(n, y)
 
        progbar.add(1, values=[("d_loss", d_loss), ("g_loss", g_loss)])
 
        # save image
#        if i % 20 == 0:
            
#            tmp = [r.reshape(-1, 28) for r in np.split(g[:100], 10)]
#            img = np.concatenate(tmp, axis = 1)
#            img = (img * 127.5 + 127.5).astype(np.uint8)
#            Image.fromarray(img).save("{}_{}.png".format(epoch, i))
# 
    # save param
    generator.save_weights(os.path.join(param_folder, 'generator_{}.hdf5'.format(epoch)))
    discriminator.save_weights(os.path.join(param_folder, 'discriminator_{}.hdf5'.format(epoch)))

#g2=g*kabukacov+kabukamean
g2=g*(Maxka-Minka)+Minka

plt.plot(kabuka[train_size:len(dataset)-1],label='true')
plt.plot(g2[0,],label='from noise')
plt.plot(g2[1,],label='from noise')
plt.plot(g2[2,],label='from noise')
plt.legend()
plt.show()

num=10.0#移動平均の個数
b=np.ones(num)/num
g3=[]
for i in range(len(g2)):       
    gm=np.convolve(g2[i,], b, mode='same')#移動平均
    g3.append(gm)

plt.plot(kabuka[train_size:len(dataset)-1],label='true')
plt.plot(g3[0],label='from noise')
plt.plot(g3[1],label='from noise')
plt.plot(g3[2],label='from noise')
plt.legend()
plt.show()