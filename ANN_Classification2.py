#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 18:45:18 2018

@author: root
"""
##Initialization
#import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas

from keras.preprocessing import sequence
from keras.models import Sequential
from keras import layers

from keras.optimizers import RMSprop
from keras.optimizers import Adagrad

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

filename0 = "H1_acom.csv"
filename1 = "H2_acom.csv"
filename2 = "H3_acom.csv"
##Constants
NBR_TST_DATA = 7
VNIR_NIR_DATA = 2500

##Main
def main():
    ##Variables
    ftrVct0 = [] #feature vector 0
    ftrVct1 = [] #feature vector 1
    rawData1 = [] #raw Data
    rawData2 = [] #raw Data
    trnData = [] #data used to train
    tstData = []  #data to test
    lbls = []
    accVct = []
    val_accVct = []
    lossVct = []
    val_lossVct = []
    epochsVct = []

##Read firts data	
    rawData0 = readFile(filename0)
    rawData1 = readFile(filename1)
    rawData2 = readFile(filename2)

    rawData = np.concatenate((rawData0[:,0:850],rawData1[:,0:850],rawData2[:,0:850]),axis = 0)
    normData = normalizeData(*rawData)
##    normData1 = normalizeData(*rawData1[:,0:850])
##    normData2 = normalizeData(*rawData2[:,0:850])
    print('Shape normData: ', np.asarray(normData).shape)
##    print('Shape normData1: ', np.asarray(normData1).shape)
##    print('Shape normData2: ', np.asarray(normData2).shape)
	
    lbls = np.concatenate((np.zeros((len(rawData0[:,0:850]),1)),np.ones((len(rawData1[:,0:850]),1)), 2*np.ones((len(rawData2[:,0:850]),1))),axis=0)
    print('Shape lbls: ', lbls.shape)
##    print('lbls: ', lbls)
##    trnData = np.concatenate((normData0,normData1,normData2),axis = 0)
##    print('Shape trnData: ', trnData.shape)
	
    x_train, x_test, y_train, y_test = train_test_split(normData, lbls, test_size=0.2, random_state=0)
    x_train = np.expand_dims(x_train,2)
    x_val = np.expand_dims(x_test,2)
    print('Shape X_train: ', np.asarray(x_train).shape)
    print('Shape Y_train: ', np.asarray(y_train).shape)
    print('Shape X_val: ', np.asarray(x_val).shape)
    print('Shape Y_val: ', np.asarray(y_test).shape)
    
##    print('Shape train_gen: ', train_gen)
    steps = range(350, 350+np.asarray(normData).shape[-1])
    print('Time steps: ', np.asarray(steps).shape)
    print('Time steps: ', steps[0])
    #plt.plot(steps,np.asarray(x_train[0]), color="g")
    #plt.show()
    #ANN building
    
    model = Sequential()
    model.add(layers.Conv1D(20, 170, activation='relu',input_shape=(850,1)))#16 #T 16,85
    model.add(layers.MaxPooling1D(2))
    #model.add(layers.Conv1D(8,15, activation = 'relu'))#16 #T16,10
##    model.add(layers.MaxPooling1D(2))
##    model.add(layers.Conv1D(32,16, activation = 'relu'))
##    model.add(layers.MaxPooling1D(2))
##    model.add(layers.Conv1D(32,16, activation = 'relu'))
##    model.add(layers.MaxPooling1D(2))
##    model.add(layers.Conv1D(32,16, activation = 'relu'))
####    model.add(layers.GlobalMaxPooling1D())
##    model.add(layers.Bidirectional(layers.GRU(32, dropout=0.1, recurrent_dropout=0.3, return_sequences=True)))
##    model.add(layers.Bidirectional(layers.GRU(32, dropout=0.1, recurrent_dropout=0.3, return_sequences=True)))
##    model.add(layers.Bidirectional(layers.GRU(64, dropout=0.1, recurrent_dropout=0.3, return_sequences=True)))
    model.add(layers.Bidirectional(layers.GRU(32, dropout=0.1, recurrent_dropout=0.3)))#16
##    model.add(layers.GRU(8, activation='relu', dropout=0.1, recurrent_dropout=0.3))
    
    
##    model.add(layers.Dense(32, activation = 'relu'))
    #model.add(layers.Flatten())
##    model.add(layers.Dense(32, activation = 'relu'))
##    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(3, activation = 'sigmoid'))
    model.summary()
    model.compile(optimizer=Adagrad(lr=0.003, epsilon=None, decay=0.0), loss='sparse_categorical_crossentropy', metrics=['acc'])
##    model.compile(optimizer=RMSprop(), loss='mae')
    #history = model.fit(x_train,y_train, epochs=2000, validation_data = (x_val, y_test))
    history = model.fit(x_train,y_train, epochs=2000, batch_size=10000000, validation_data = (x_val, y_test))
##    history = model.fit_generator(train_gen, steps_per_epoch = 10,epochs=100,validation_data=val_gen, validation_steps = 100)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

##    accVct.append(acc)
##    val_accVct.append(val_acc)
##    lossVct.append(loss)
##    val_lossVct.append(val_loss)
##    epochsVct = []
    
    df = pandas.DataFrame(data={"epochs": epochs, "loss": loss, "val_loss": val_loss,"acc": acc,"val_acc": val_acc})
    df.to_csv("./Results_2.csv", sep=',',index=False)

    #model.save_weights('my_model_weights.h5')
    #model.save('my_model.h5')
    
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    
    
	
############################
# FUNCTIONS
############################
#Data generator
def dataGenerator(x_train, y_train):
    yield x_train, y_train

#Normalize Data
def normalizeData(*rawData):
    [rows, cols] = np.asarray(rawData).shape
    normData = []
    for i in range(0, rows):
        tmpVct = rawData[i]
        max_abs_scaler = preprocessing.MaxAbsScaler()
        tmpVct = max_abs_scaler.fit_transform(tmpVct.reshape(-1,1))
        #print('Shape rawData: ', np.asarray(tmpVct).shape)
        normData.append(tmpVct[:,0])
    return normData
	

#Read files
def readFile(filename):
    data = []
    outVct = []
    f = open(filename,"r")

    ctr0 = 0
    for line in f:
        data.append(line)
        outVct.append(data[ctr0].split(","))
        ctr0 +=1
    for i in range(0, len(outVct)):
        for j in range(0, len(outVct[0])):
            outVct[i][j] = float(outVct[i][j])
    outVct = np.transpose(outVct)
    return outVct[:,0:VNIR_NIR_DATA]

if __name__ == '__main__':
    main()
