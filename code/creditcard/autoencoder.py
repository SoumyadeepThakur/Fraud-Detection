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
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from confusion import evaluate
from select_feature import return_k_best
from sklearn import linear_model, datasets, metrics
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
	
	# Prepare for train
	X_train, X_test = train_test_split(data, test_size=0.3, random_state=RANDOM_SEED, stratify=data['Class'].values)
	
	#Scale data. Fit scaler on "train" data and scale both "train" and "test" data
	scaler = MinMaxScaler()
	scaler.fit(X_train)
	X_train = pd.DataFrame(scaler.transform(X_train), columns = data.columns)
	X_test = pd.DataFrame(scaler.transform(X_test), columns = data.columns)

	Y_train = X_test['Class']
	X_train = X_train.drop(['Class'], axis=1)
	Y_test = X_test['Class']
	X_test = X_test.drop(['Class'], axis=1)

	# Convert pandas dataframe into numpy array (convenience, nothing important)

	X_train = X_train.values
	X_test = X_test.values
	Y_train = Y_train.values
	Y_test = Y_test.values
	
	return X_train, Y_train, X_test, Y_test
	

def train_model(model, X_train, batch_size = 32, epoch = 100, load=False):

	with tf.Session() as session:
		path_model = './model_auto' # model file
		path_logdir = 'logs_auto' # log files

		if load == True: # Load the previously trained model 
			saver = tf.train.Saver()
			saver.restore(session, path_model)
		else:
			init = tf.global_variables_initializer()
			session.run(init)


		saver = tf.train.Saver()
		writer = tf.summary.FileWriter(path_logdir, session.graph)
		
		
		X_train = np.array(X_train)

		for i in range(epoch):
			for count in range(0,(X_train.shape[0]//batch_size)*batch_size,batch_size): # extract batch_size transactions at a time (batch size = batch_size)
				
				in_vector = X_train[count:count+batch_size]
				feed = {model['ip']: in_vector} # feed the input vector to placeholder

				# execute the tensorflow training session. run model['optimizer'] because that is defined as the tensor that minimizes the model['cost']
				_, summary = session.run([model['optimizer'], model['sum_loss']], feed_dict = feed)
			writer.add_summary(summary, i)

			print('Epoch: ', i)

			np.random.shuffle(X_train)

		saver.save(session, path_model)


def test_model(model, X_test, Y_test, batch_size = 32):

	path_model = './model_auto'

	# write output of autoencoders to file
	with open('logfile-autoencoder.txt','w+') as outfile:
		with tf.Session() as sess:

			# Load tensorflow model from file
			saver = tf.train.Saver()
			saver.restore(sess, path_model)

			for i in range(0,(X_test.shape[0]//batch_size)*batch_size,batch_size): # extract batch_size transactions at a time (batch size = batch_size)

				in_vector = X_test[i:i+batch_size]
				feed = {model['ip']: in_vector}

				outfile.write('Batch --- ' + str(i//batch_size) +' --- \n')
				ans = sess.run(model['cost-2'], feed_dict =  feed)
				outfile.write(str(list(ans)))
				outfile.write('\n')



def consfusion_eval(labels, file):

	# This function calculates the Precision and Recall for different values of threshold

	TPX, TNX, FPX, FNX = list(), list(), list(), list()
	for i in range(400,600):
		tp, tn, fp, fn = evaluate(labels, file, (i/10000))
		print('Thresh: ', i/10000, 'Conf: ', tp, ' -- ', tn, ' -- ', fp, ' -- ', fn)
		acc=(tp+tn)/(tp+tn+fp+fn)
		sens=tn/(tn+fp)
		spec=tp/(tp+fn)
		prec=tn/(tn+fn)
		print('Acc: ',acc, ' Sens: ', sens, ' Spec: ', spec, ' Prec: ', prec)


def main():
	
	X_train, Y_train, X_test, Y_test = make_data()
	model = autoencoder_model(input_shape=X_train.shape[1])
	train_model(model, X_train, epoch = 250, load=False)
	test_model(model, X_test, Y_test)
	consfusion_eval(Y_test, 'logfile-autoencoder.txt')

if __name__ == "__main__":
	main()
