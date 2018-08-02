#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 00:32:40 2018

@author: titoarevaloramirez
"""

##Initialization
from __future__ import division
import time

import pywt
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
from sklearn import svm
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor


filename0 = "/home/usr3/Documents/UTFSM/1erSemestre/MachineLearning/Proyecto/Data/Hojas1.txt"

filename1 = "/home/usr3/Documents/UTFSM/1erSemestre/MachineLearning/Proyecto/Data/Hojas2.txt"

##Constants
VNIR_NIR_DATA = 500
TRAIN_LENGTH = 120
RF_DEPTH = 60
RF_STATE = 2

##Main
def main():

	##Variables
	ftrVct0 = [] #feature vector 0
	ftrVct1 = [] #feature vector 1
	rawData = [] #raw Data
	trnData = [] #data used to train
	tstData = []  #data to test
	lbls = []

##Read firts data	
	rawData_H1 = readFile(filename1)
	rows_H1, cols_H1 = rawData_H1.shape
	vnirNir_H1 = rawData_H1[:,0:(VNIR_NIR_DATA-1)]
	swir_H1 = rawData_H1[:, VNIR_NIR_DATA:cols_H1]
	ftrVct0 = featureExtraction(*vnirNir_H1)

	for k in range(121,199):
		espectraTest = k
		regr_rf = RandomForestRegressor(max_depth=RF_DEPTH, random_state=RF_STATE)
		predictedEspectra = np.zeros((cols_H1-VNIR_NIR_DATA,1))
		for i in range(cols_H1-VNIR_NIR_DATA):
			regr_rf.fit(ftrVct0[0:TRAIN_LENGTH,:], swir_H1[0:TRAIN_LENGTH,i]) 
	#        a = ftrVct0[1,:]
	#        a = a.reshape(1,-1)
			predcited_rf = regr_rf.predict(ftrVct0[espectraTest,:].reshape(1,-1))
			predictedEspectra[i] = predcited_rf
		
		fig = plt.figure()
		fig.suptitle('Espectral Signature', fontsize=14, fontweight='bold')
		ax = fig.add_subplot(111)
		fig.subplots_adjust(top=0.85)
		#ax.set_title('')
		ax.set_xlabel('X [nm]')
		ax.set_ylabel('Y []')
		plt.grid(True)
		idealEspectra = np.asarray(rawData_H1[espectraTest][0:cols_H1])
		print('Shape Ideal Espectra: ' + str(idealEspectra.shape))
		print('Shape predicted Espectra: ' + str(predictedEspectra.shape))
		totalEspectra = np.concatenate((idealEspectra[0:VNIR_NIR_DATA].reshape((VNIR_NIR_DATA,1)), predictedEspectra), axis=0)
		ideal = plt.plot(idealEspectra, color="g")
		predicted = plt.plot(totalEspectra, color="r")
			
		plt.legend([ideal, predicted], ["Ideal", "Predicted"])
		
		filename = 'hoja' + str(espectraTest);
		#fig.savefig(filename+'.svg', format='svg', dpi=1200)
		np.savetxt(filename + '.out', totalEspectra, delimiter=',')   # X is an array
	plt.show()

############################
# FUNCTIONS
############################

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
	return outVct

#Extract Features
def featureExtraction(*outVct):
	##Local Variables
	detailVct = []	
	statFeatVct = []
	##Subband descomposition
	for i in range(0, len(outVct)):
		(cA1, cD1) = pywt.dwt(outVct[i],'db3')
		(cA2, cD2) = pywt.dwt(cA1,'db3')
		(cA3, cD3) = pywt.dwt(cA2,'db3')
		(cA4, cD4) = pywt.dwt(cA3,'db3')
		(cA5, cD5) = pywt.dwt(cA4,'db3')
		detailVct.append([cD1,cD2,cD3,cD4,cD5,cA5])
	
	##Statistical Features
	for i in range(0, len(outVct)):
		cD1mean = np.mean(np.absolute(detailVct[i][0]))
		cD1AvPow = np.mean(np.power(detailVct[i][0],2)/np.size(detailVct[i][0]))
		cD1Std = np.std(detailVct[i][0])
		#
		cD2mean = np.mean(np.absolute(detailVct[i][1]))
		cD2AvPow = np.mean(np.power(detailVct[i][1],2)/np.size(detailVct[i][1]))
		cD2Std = np.std(detailVct[i][1])
		#	
		cD3mean = np.mean(np.absolute(detailVct[i][2]))
		cD3AvPow = np.mean(np.power(detailVct[i][2],2)/np.size(detailVct[i][2]))
		cD3Std = np.std(detailVct[i][2])
		#
		cD4mean = np.mean(np.absolute(detailVct[i][3]))
		cD4AvPow = np.mean(np.power(detailVct[i][3],2)/np.size(detailVct[i][3]))
		cD4Std = np.std(detailVct[i][3])
		#
		cD5mean = np.mean(np.absolute(detailVct[i][4]))
		cD5AvPow = np.mean(np.power(detailVct[i][4],2)/np.size(detailVct[i][4]))
		cD5Std = np.std(detailVct[i][4])
		#
		cA5mean = np.mean(np.absolute(detailVct[i][5]))
		cA5AvPow = np.mean(np.power(detailVct[i][5],2)/np.size(detailVct[i][5]))
		cA5Std = np.std(detailVct[i][5])
		#
		cD1ratio = cD1mean/cD2mean
		cD2ratio = cD2mean/cD3mean
		cD3ratio = cD3mean/cD4mean
		cD4ratio = cD4mean/cD5mean
		cD5ratio = cD5mean/cD4mean
		#
		statFeatVct.append([cD1mean, cD1AvPow, cD1Std, cD1ratio, cD2mean, cD2AvPow, cD2Std, cD2ratio, cD3mean, cD3AvPow, cD3Std, cD3ratio, cD4mean, cD4AvPow, cD4Std, cD4ratio, cD5mean, cD5AvPow, cD5Std, cD5ratio, cA5mean, cA5AvPow, cA5Std])
		#
	max_abs_scaler = preprocessing.MaxAbsScaler()
	return max_abs_scaler.fit_transform(statFeatVct) ##Return feature vector scaled

if __name__ == '__main__':
	main()

#print(cA)
#print(cD)
#plt.subplot(231)
#plt.plot(cA5)
#plt.title('CA5')
#plt.subplot(232)
#plt.plot(cD1)
#plt.title('CD1')
#plt.subplot(233)
#plt.plot(cD2)
#plt.title('CD2')
#plt.subplot(234)
#plt.plot(cD3)
#plt.title('CD3')
#plt.subplot(235)
#plt.plot(cD4)
#plt.title('CD4')
#plt.subplot(236)
#plt.plot(cD5)
#plt.title('CD5')
#plt.show()
##plt.plot(cA, cD)
##plt.ylabel('wavelets')
##plt.show()
