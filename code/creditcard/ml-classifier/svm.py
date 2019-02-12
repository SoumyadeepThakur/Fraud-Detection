import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm

def make_data():
	RANDOM_SEED = 42
	df = pd.read_csv('creditcard.csv')
	data = df
	data['Time'] = data['Time'].apply(lambda x : ((x//60)%1440)/1440) # convert to time of day in minuites
	cols = data.columns.values
	for col in cols[1:len(cols)]:
		data[col] = MinMaxScaler().fit_transform(data[col].values.reshape(-1, 1))	
	data['Class'] = data['Class'].apply(lambda x : int(x//1))
	# Prepare for train

	X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
	Y_train = X_train['Class']
	X_train = X_train.drop(['Class'], axis=1)
	Y_test = X_test['Class']
	X_test = X_test.drop(['Class'], axis=1)
	X_train = X_train.values
	Y_train = Y_train.values
	X_test = X_test.values
	Y_test = Y_test.values

	return X_train, Y_train, X_test, Y_test



def main():

	verbose=1
	X_train, Y_train, X_test, Y_test = make_data()
	svm_clf = svm.SVC(gamma=0.002)
	svm_clf.fit(X_train, Y_train)

	pred = svm_clf.predict(X_test)
	tp = tn = fp = fn =0
	for i in range(len(Y_test)):
		if Y_test[i]==0:
			if pred[i]==0:
				tp+=1
			else:
				fn+=1
		if Y_test[i]==1:
			if pred[i]==1:
				tn+=1
			else:
				fp+=1

	print(tp, ' --- ', tn, ' --- ', fp, ' --- ', fn)

main()


