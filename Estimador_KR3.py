from __future__ import division
import time

import numpy as np


import pywt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt




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
#    return statFeatVct
	return max_abs_scaler.fit_transform(statFeatVct) ##Return feature vector scaled

# son 199 hojas las medidas

#filename0 = "H1_acom.csv"
filename1 = "H3_acom.csv"


final = 1200
med = final-350   # Rango de longitud de onda para el entrenamiento
len_train=119           # Cantidad de hojas para el entrenamiento
len_test = 199-len_train
 

rawData = readFile(filename1)
rawData_H1 = rawData
rawData_C1 = rawData_H1[:,0:850] # Solo caract. importantes 
ftrVct0 = featureExtraction(*rawData_C1)



# =============================================================================
# Estimador
# =============================================================================

kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})

#t0 = time.time()
#pm = np.zeros((2500-med-349,1)) # Prediccion creo que esta bn el tama
#aa = np.zeros((2500-med-349,1)) # 351

alls = np.zeros((1301,78))# o 1301,78
print(len(range(850,2150)))

for i in range(850,2151): # desde 850
    kr.fit(ftrVct0[0:120,:], rawData_H1[0:120,i])
    print(i)
    for j in range(121,199): # 120,198    
        sdds = kr.predict(ftrVct0[j,:].reshape(1,-1))
        alls[i-850,j-121] = sdds#con 120
        
        
for j in range(120,198):
    b=rawData[j+1,0:850]   # PARA GUARDAR SIN EL GROUNDTRUTH, podria ser 1 mas SE LE AUMENTO EL +1
#    b=rawData[j+len_train,:]      # PARA GUARDAR TODO Y GROUNDTRUTH
    a=np.concatenate((b,alls[:,j-120]),axis=0)
    plt.plot(alls[:,j-120], color="r")
    # Guardar datos
    filet = 'foc'+str(j)+'.out'
    np.savetxt(filet, a, delimiter=",")

    
plt.xlabel("Longitud de onda")
plt.ylabel("Reflectancia")
plt.title('Predicción')
#plt.legend(loc="best")
plt.savefig('intento.eps', format='eps', dpi=1000)


plt.show()

