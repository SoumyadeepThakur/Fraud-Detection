'''
AUTHOR: SOUMYADEEP THAKUR
DATE: 6 OCT 2018
'''

import numpy as np
import signal
import sys
import math
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from confusion import evaluate
from select_feature import return_k_best
from  sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import SMOTE
from models.models import autoencoder_model


'''
TO DO: DROP TIME ATTRIBUTE AND TRY
'''

def make_data():
	RANDOM_SEED = 42
	df = pd.read_csv('creditcard.csv')
	data = df
	data['Time'] = data['Time'].apply(lambda x : ((x//60)%1440)/1440) # convert to time of day in minuites
	#data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
	
	# Select features
	#data = return_k_best(data, k_features=11)
	# Prepare for train
	scaler = MinMaxScaler()
	X_train, X_test = train_test_split(data, test_size=0.3, random_state=RANDOM_SEED)
	scaler.fit(X_train)
	X_train = pd.DataFrame(scaler.transform(X_train), columns = data.columns)
	X_test = pd.DataFrame(scaler.transform(X_test), columns = data.columns)
	Y_train = X_train['Class']
	X_train = X_train.drop(['Class'], axis=1)
	Y_test = X_test['Class']
	X_test = X_test.drop(['Class'], axis=1)
	X_train = X_train.values
	Y_train = Y_train.values
	X_test = X_test.values
	Y_test = Y_test.values

	#print(X_train, X_test)
	return X_train, Y_train, X_test, Y_test


def get_hidden(X_train, X_test):

	path_model = './model_auto'
	model = autoencoder_model(X_train.shape[1])
	encodings_train = list()
	encodings_test = list()

	with tf.Session() as sess:

		saver = tf.train.Saver()
		saver.restore(sess, path_model)

		for i in range(0,(X_train.shape[0]//32)*32,32):
			if i % 65536 == 0:
				print('Encoding', i//1024)
			in_vector = X_train[i:i+32]
			feed = {model['ip']: in_vector}

			ans = np.array(sess.run(model['ye_2'], feed_dict =  feed))
			#print(list(ans))
			for j in range(32):
				encodings_train.append(ans[j])

		for i in range(0,(X_test.shape[0]//32)*32,32):
			if i % 65536 == 0:
				print('Encoding', i//1024)
			in_vector = X_test[i:i+32]
			feed = {model['ip']: in_vector}

			ans = np.array(sess.run(model['ye_2'], feed_dict =  feed))
			#print(list(ans))
			for j in range(32):
				encodings_test.append(ans[j])

	return encodings_train, encodings_test

def train_mlp(X_train, Y_train):

	clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(13, 7), n_iter_no_change=40, shuffle=True, tol=5e-5, solver='adam', batch_size=32, learning_rate='adaptive', max_iter=100, verbose=True)
	print('Fitting model ...')
	clf.fit(X_train, Y_train)
	print('Fitting done')
	return clf

def sample(X_train, Y_train):
	

	# Declare how much to sample from each class
	sampling_dict = {0: 120000}
	print('Undersampling ...')
	rus = RandomUnderSampler(random_state=113, sampling_strategy=sampling_dict)
	#tkl = TomekLinks(sampling_strategy=sampling_dict, n_jobs=-1)
	X_train, Y_train = rus.fit_resample(X_train, Y_train)

	# Apply SMOTE
	# To reduce class imbalance perform SMOTE
	print('Performing SMOTE....')
	smote_dict = {1: 12000}
	sm = SMOTE(random_state=42, sampling_strategy=smote_dict, n_jobs=-1)
	X_train, Y_train = sm.fit_sample(X_train, Y_train)

	return X_train, Y_train

def test_mlp(X_test, Y_test, clf):

	y_pred = clf.predict(X_test)
	tp = tn = fp = fn =0
	for i in range(len(X_test)):
		if Y_test[i]==0:
			if y_pred[i]==0:
				tp+=1
			else:
				fn+=1
		else:
			if y_pred[i] != 0:
				tn+=1
			else:
				fp+=1

	print(tp, ' --- ', tn, ' --- ', fp, ' --- ', fn)


def main():
	
	X_train, Y_train, X_test, Y_test = make_data()
	enc_train, enc_test = get_hidden(X_train, X_test)
	enc_train = np.array(enc_train)
	enc_test = np.array(enc_test)
	Y_train = Y_train[0:len(enc_train)]
	Y_test = Y_test[0:len(enc_test)]
	print('done')
	#enc_train, Y_train = sample(enc_train, Y_train)
	print(enc_train.shape)
	print(Y_train.shape)
	print(enc_test.shape)
	print(Y_test.shape)
	clf = train_mlp(enc_train, Y_train)
	test_mlp(enc_test, Y_test, clf)

main()