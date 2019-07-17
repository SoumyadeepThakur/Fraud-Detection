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
	
'''
def create_model(input_shape,  lrate = 1e-5):

	model = dict()

	dim_input = input_shape
	dim_enc_1 = 22 # First hidden layer dimension
	dim_hidden = 15 # Second hidden layer dimension
	dim_dec_1 = 22
	dim_dec_2 = dim_input

	#create model
	with tf.name_scope('input'):
		model['ip'] = tf.placeholder(tf.float32, [None, dim_input], name = 'input-vector') # Input vector will be fed to this tensor

	# FIRST ENCODING LAYER

	with tf.name_scope('encoding-layer-1'):

		# Code ye_1 = leaky_relu(We_1*ip + Be_1) using tensors

		model['We_1'] = tf.Variable(tf.random_normal([dim_input, dim_enc_1], stddev=1.0/dim_enc_1), name = 'We-1')
		model['Be_1'] = tf.Variable(tf.random_normal([1, dim_enc_1], stddev=1.0/dim_enc_1), name = 'Be-1')
		model['ye_1'] = tf.nn.leaky_relu(tf.add(tf.matmul(model['ip'], model['We_1']), model['Be_1']), alpha=0.1, name = 'ye-1')

	# SECOND ENCODING LAYER

	with tf.name_scope('encoding-layer-2'):

		# Code ye_2 = leaky_relu(We_2*ye_1 + Be_2) using tensors

		model['We_2'] = tf.Variable(tf.random_normal([dim_enc_1, dim_hidden], stddev=1.0/dim_hidden), name = 'We-2')
		model['Be_2'] = tf.Variable(tf.random_normal([1, dim_hidden], stddev=1.0/dim_hidden), name = 'Be-2')
		model['ye_2'] = tf.nn.leaky_relu(tf.add(tf.matmul(model['ye_1'], model['We_2']), model['Be_2']), alpha=0.1, name = 'ye-2')

	
	# FIRST DECODING LAYER

	with tf.name_scope('decoding-layer-1'):

		# Code yd_1 = leaky_relu(Wd_1*ye_2 + Bd_1) using tensors

		model['Wd_1'] = tf.Variable(tf.random_normal([dim_hidden, dim_dec_1], stddev=1.0/dim_dec_1), name = 'Wd-1')
		model['Bd_1'] = tf.Variable(tf.random_normal([1, dim_dec_1], stddev=1.0/dim_dec_1), name = 'Bd-1')
		model['yd_1'] = tf.nn.leaky_relu(tf.add(tf.matmul(model['ye_2'], model['Wd_1']), model['Bd_1']), alpha=0.1, name = 'yd-1')

	# SECOND DECODING LAYER

	with tf.name_scope('decoding-layer-2'):
		model['Wd_2'] = tf.Variable(tf.random_normal([dim_dec_1, dim_dec_2], stddev=1.0/dim_dec_2), name = 'Wd-2')
		model['Bd_2'] = tf.Variable(tf.random_normal([1, dim_dec_2], stddev=1.0/dim_dec_2), name = 'Bd-2')

	

	with tf.name_scope('output'):

		# Code op = tanh(Wd_2*yd_1 + Bd_2) using tensors

		model['op'] = tf.nn.tanh(tf.add(tf.matmul(model['yd_1'], model['Wd_2']), model['Bd_2']), name = 'output-vector')

	# Loss metrics and optimizers

	with tf.name_scope('loss_optim_4'):

		# LSE error considered for optimization
		model['cost'] = tf.reduce_mean(tf.squared_difference(model['ip'], model['op']), name = 'cost')
		model['cost-2'] = tf.reduce_sum(tf.squared_difference(model['ip'], model['op']), axis=1, name='cost-2')
		model['optimizer'] = tf.train.AdamOptimizer(lrate).minimize(model['cost'], name = 'optim')
		model['sum_loss'] = tf.summary.scalar(model['cost'].name, model['cost']) # for logging purposes

	# return the model dictionary

	return model;

'''
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