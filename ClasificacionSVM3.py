
import pywt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix

import pickle



filename0 = "/home/usr3/Documents/UTFSM/1erSemestre/MachineLearning/Proyecto/Data/Hojas1.txt"
filename1 = "/home/usr3/Documents/UTFSM/1erSemestre/MachineLearning/Proyecto/Data/Hojas2.txt"
filename2 = "/home/usr3/Documents/UTFSM/1erSemestre/MachineLearning/Proyecto/Data/Hojas3.csv"


##Constants
NBR_TST_DATA = 7
lol=[]

##Main
def main():
 
	##Variables
	ftrVct0 = [] #feature vector 0
	ftrVct1 = [] #feature vector 1
	rawData = [] #raw Data
	trnData = [] #data used to train
	tstData = []  #data to test
	lbls = []
# =============================================================================
#     IMPORTANTE PARA CLASIFICAR
# =============================================================================
    
##Read firts data	
	rawData0 = readFile(filename0)
	rawData0 = rawData0[:,0:850] 	                # Recortando las bandas
	ftrVct0 = featureExtraction(*rawData0)			# 23 features
##	ftrVct0 = rawData0 				 #### VA CON TODOS LOS DATOS (TODAS LAS BANDAS)
    
##Read second data

	rawData1 = readFile(filename1)
	rawData1 = rawData1[:,0:850]
	ftrVct1 = featureExtraction(*rawData1)
##	ftrVct1 = rawData1 				 #### VA CON TODOS LOS DATOS
  
## Read Third data
	rawData2 = readFile(filename2)
	rawData2 = rawData2[:,0:850]
	ftrVct2 = featureExtraction(*rawData2)
##	ftrVct2 = rawData2 				 #### VA CON TODOS LOS DATOS	
	
	rawData = np.concatenate((ftrVct0,ftrVct1,ftrVct2),axis = 0)
	trnData = normalizeData(*rawData)
    

	
	lbls = np.concatenate((np.zeros((len(ftrVct0),1)),np.ones((len(ftrVct1),1)), 2*np.ones((len(ftrVct2),1))),axis=0)

	
	X_train, X_test, y_train, y_test = train_test_split(trnData, lbls, test_size=0.2, random_state=0)
	y_true = y_test; # Almacenando los labels anteriores
	
	#C_range = np.logspace(-2, 10, 20)
	#gamma_range = np.logspace(-9, 3, 20)
	#param_grid = dict(gamma=gamma_range, C=C_range)
	#cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
	#grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
	#grid.fit(X_train,y_train.ravel())
	
	# fijo
	grid = SVC(kernel='rbf', random_state=0, gamma=1, C=1)
	grid.fit(X_train,y_train.ravel())
	
	
#	C = grid.best_estimator_.C
#	print("Best C: %0.5f"%C)
#	g = grid.best_estimator_.gamma
#	print("Best gamma: %0.5f"%g)
	trnScore = grid.score(X_train, y_train.ravel())
	print("Score (trainData): %0.5f"%trnScore)
	tstScore = grid.score(X_test, y_test.ravel())
	print("Score (testData): %0.5f"%tstScore)
	tstPredict = grid.predict(X_test)
	a = confusion_matrix(y_true,tstPredict)
	print(a)
# =============================================================================
# GUARDAR EL CLASIFICADOR
# =============================================================================
# now you can save it to a file
	with open('filename.pkl', 'wb') as f:
		pickle.dump(grid, f)
# =============================================================================
# LEER EL CLASIFICADOR
# =============================================================================

#	with open('filename.pkl', 'rb') as f:
#		clf = pickle.load(f)
    
	
	
	#clf = svm.SVC(C=float(C), kernel = 'rbf', gamma = float(g))
	#clf.fit(trnData,lbls.ravel()) 
#	predicted0 =  grid.predict(X_train)
#	predicted1 =  grid.predict(X_test)
	
	#title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
	# SVC is more expensive so we do a lower number of CV iterations:
	#estimator = svm.SVC(C=float(C), kernel = 'rbf', gamma = float(g))
	#plot_learning_curve(estimator, title, trnData, lbls.ravel(), (0.7, 1.01), cv=cv, n_jobs=4)

	#plt.show()

#Lbls for all data	
	#lbls = np.concatenate((np.zeros((len(ftrVct0),1)),np.ones((len(ftrVct1),1))),axis=0)
	#fullData = np.concatenate((ftrVct0,ftrVct1),axis = 0)	
	#predicted = cross_val_predict(clf, fullData, lbls, cv=10)	

#	for i in range(0,int(len(ftrVct0)+len(ftrVct1))):
#		if grid.predict(trnData[i].reshape(1,-1)) == 0:
#			print('Es la hoja 1', 'Deberia ser:', lbls[i]+1)
#		else:
#			print('Es la hoja 2', 'Deberia ser:', lbls[i]+1)

############################
# FUNCTIONS
############################

##Plot Learning Functions
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, 
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


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
#		print(cA5)
    
	
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
