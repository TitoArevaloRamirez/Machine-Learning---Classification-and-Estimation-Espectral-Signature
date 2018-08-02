#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 18:45:18 2018

@author: root
"""
##Initialization
import numpy as np
import matplotlib.pyplot as plt
import pandas

from keras.preprocessing import sequence
from keras.models import Sequential
from keras import layers
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

filename0 = "/home/usr3/Documents/UTFSM/1erSemestre/MachineLearning/Proyecto/Data/Hojas1.txt"
filename1 = "/home/usr3/Documents/UTFSM/1erSemestre/MachineLearning/Proyecto/Data/Hojas2.txt"
filename2 = "/home/usr3/Documents/UTFSM/1erSemestre/MachineLearning/Proyecto/Data/Hojas3.csv"
##Constants
VNIR_NIR_DATA = 2500
VNIR_DATA = 850
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
    
##Read firts data	
    rawData0 = readFile(filename0)
    print('Cols normData: ', np.asarray(rawData0).shape[1])
    rawData1 = readFile(filename1)
    rawData2 = readFile(filename2)

##    rawData = np.concatenate((rawData0[:,0:850],rawData1[:,0:850],rawData2[:,0:850]),axis = 0)
    #Select tipe of tree
    normData = rawData0
    print('Shape normData: ', normData.shape)
    
	
##    lbls = np.concatenate((np.zeros((len(rawData0[:,0:850]),1)),np.ones((len(rawData1[:,0:850]),1)), 2*np.ones((len(rawData2[:,0:850]),1))),axis=0)
##    for i in range(VNIR_DATA, np.asarray(rawData0).shape[1]): #np.asarray(normData).shape[1]
##    lbls = np.ones((len(rawData0[:,0:VNIR_DATA]),1))*np.reshape(rawData0[:,VNIR_DATA:3], (len(rawData0[:,0:VNIR_DATA]),1))
    
##    print('Shape lbls: ', lbls.shape)
##    x_train, x_test, y_train, y_test = train_test_split(normData, lbls, test_size=0.2, random_state=0)
    x_train = normData[0:159,0:VNIR_DATA]
    x_train = np.expand_dims(x_train,2)
    x_val = normData[159:199,0:VNIR_DATA]
    x_val = np.expand_dims(x_val,2)
    y_train = normData[0:159,VNIR_DATA:np.asarray(rawData0).shape[1]]
    y_test = normData[159:199,VNIR_DATA:np.asarray(rawData0).shape[1]]
    print('Shape X_train: ', np.asarray(x_train).shape)
    print('Shape Y_train: ', np.asarray(y_train).shape)
    print('Shape X_val: ', np.asarray(x_val).shape)
    print('Shape Y_val: ', np.asarray(y_test).shape)
       
    seed = 3
    np.random.seed(seed)
    model = Sequential()
    model.add(layers.Conv1D(1301, 850, activation='relu',input_shape=(VNIR_DATA,1)))
##    model.add(layers.MaxPooling1D(2))
##    model.add(layers.Conv1D(16, 10, activation='relu',input_shape=(VNIR_DATA,1)))
##    model.add(layers.MaxPooling1D(2))
    model.add(layers.Bidirectional(layers.GRU(1301, dropout=0.1, recurrent_dropout=0.1)))
    model.add(layers.Dense(1301, activation = 'sigmoid'))
    model.summary()
    model.compile(optimizer=Adagrad(lr=0.01, epsilon=None, decay=0.0), loss='logcosh')#, metrics = ['mae'])
    history = model.fit(x_train,y_train, epochs=600, validation_data = (x_val, y_test))
    filename_save = "modelRegressionTest" + str(1) + ".h5" 
    model.save(filename_save)
        
##    mae = history.history['mean_absolute_error']
##    val_mae = history.history['val_mean_absolute_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
####
##    df = pandas.DataFrame(data={"epochs": epochs, "loss": loss, "val_loss": val_loss,"mae": mae,"val_mae": val_mae})
##    filename_save = "./resultsRegressionH" + str(1) + ".csv"
##    df.to_csv(filename_save, sep=',',index=False)
        
  
    plt.close('all')
##    plt.plot(epochs, mae, 'r', label='Training mae')
##    plt.plot(epochs, val_mae, 'b', label='Validation mae')
##    plt.title('Training and validation accuracy')
##    plt.legend()
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
    return np.asarray(normData)
	

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
