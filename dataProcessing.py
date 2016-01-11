import pandas
import numpy as np

def get_data_train(fileName):
	data = pandas.read_csv(fileName)
	xtr = data.iloc[:,1:].values
	xtr = [np.reshape(i,(784,1)) for i in xtr]
	ytr = []
	for digit in data['label'].values:
		ytemp = np.zeros(10)
		ytemp[digit] = 1.
		ytr.append(ytemp)
	ytr = np.array(ytr)
	ytr = [np.reshape(i,(10,1)) for i in ytr]
	output = zip(xtr,ytr)
	return output

def get_data_test(fileName):
	data = pandas.read_csv(fileName)
	xtr = data.iloc[:,1:].values
	xtr = [np.reshape(i,(784,1)) for i in xtr]
	ytr =  data['label'].values
	ytr = np.array(ytr)
	output = zip(xtr,ytr)
	return output
	
def getImg(tableau):
	img = np.zeros((28,28))
	for i in range(0,27):
		for j in range(0,27):
			index = i*28 + j
			img[i][j] = tableau[index]
	return img
