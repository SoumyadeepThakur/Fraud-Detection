import sys
import numpy as np
from collections import Counter
from dbscan_predict import DBSCAN_Predict
from preprocess import preprocess_data

def confusion(output, label):

	assert len(output) == len(label)

	tp = tn = fp = fn =0
	for i in range(len(output)):
		if label[i]==0:
			if output[i]==0:
				tp+=1
			else:
				fn+=1
		else:
			if output[i] != 0:
				tn+=1
			else:
				fp+=1

	print(tp, ' --- ', tn, ' --- ', fp, ' --- ', fn)

def main():

	verbose=1
	X, Y = preprocess_data('creditcard.csv')
	X_train=X[0:32768]
	X_test=X[32768:65536]
	Y_train=Y[0:32768]
	Y_test=Y[32768:65536]
	
	# Train

	dbscan = DBSCAN_Predict(eps=0.23, min_samples=3, n_jobs=4)
	pred = dbscan.fit_predict(X_train)

	classes = dict()
	for i in range(len(X_train)):
		p=int(pred[i])
		if p not in classes: classes[p] = 0
		classes[p]=classes[p]+1

	for x in classes: 
		print('class', x, classes[x])

	confusion(pred, Y_train)
	
	# Predict

	y_new = dbscan.predict(X_test)

	confusion(y_new, Y_test)

if __name__ == "__main__":

	main()


