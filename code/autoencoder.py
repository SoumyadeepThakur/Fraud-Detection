'''
AUTHOR: SOUMYADEEP THAKUR
DATE: 6 OCT 2018
'''

import numpy as np
import signal
import sys
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

'''
TO DO: DROP TIME ATTRIBUTE AND TRY
'''

def make_data():
	RANDOM_SEED = 42
	df = pd.read_csv('creditcard.csv')
	data = df
	data['Time'] = data['Time'].apply(lambda x : ((x//60)%1440)/1440) # convert to time of day in minuites
	data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
	# Prepare for train
	X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
	X_train = X_train[X_train.Class == 0]
	X_train = X_train.drop(['Class'], axis=1)
	Y_test = X_test['Class']
	X_test = X_test.drop(['Class'], axis=1)
	X_train = X_train.values
	X_test = X_test.values
	Y_test = Y_test.values

	#print(X_train, X_test)
	return X_test, X_train, Y_test
	

def create_model(X_test, X_train, Y_test,  lrate = 1e-5):

	#initialize data
	#X_test, X_test, Y_test = make_data()

	#initialize parameters
	model = dict()

	dim_b = 32
	dim_input = X_train.shape[1]
	dim_enc_1 = 14
	dim_enc_2 = 7
	dim_dec_1 = 14
	dim_dec_2 = dim_input

	#create model
	with tf.name_scope('input'):
		model['ip'] = tf.placeholder(tf.float32, [dim_b, dim_input], name = 'input-vector')

	# FIRST ENCODING LAYER

	with tf.name_scope('encoding-layer-1'):
		model['We_1'] = tf.Variable(tf.random_normal([dim_input, dim_enc_1], stddev=1.0/dim_enc_1), name = 'We-1')
		model['Be_1'] = tf.Variable(tf.random_normal([1, dim_enc_1], stddev=1.0/dim_enc_1), name = 'Be-1')
		model['ye_1'] = tf.nn.tanh(tf.add(tf.matmul(model['ip'], model['We_1']), model['Be_1']), name = 'ye-1')

	# SECOND ENCODING LAYER

	with tf.name_scope('encoding-layer-2'):
		model['We_2'] = tf.Variable(tf.random_normal([dim_enc_1, dim_enc_2], stddev=1.0/dim_enc_2), name = 'We-2')
		model['Be_2'] = tf.Variable(tf.random_normal([1, dim_enc_2], stddev=1.0/dim_enc_2), name = 'Be-2')
		model['ye_2'] = tf.nn.relu(tf.add(tf.matmul(model['ye_1'], model['We_2']), model['Be_2']), name = 'ye-2')

	# FIRST DECODING LAYER

	with tf.name_scope('decoding-layer-1'):
		model['Wd_1'] = tf.Variable(tf.random_normal([dim_enc_2, dim_dec_1], stddev=1.0/dim_dec_1), name = 'Wd-1')
		model['Bd_1'] = tf.Variable(tf.random_normal([1, dim_dec_1], stddev=1.0/dim_dec_1), name = 'Bd-1')
		model['yd_1'] = tf.nn.tanh(tf.add(tf.matmul(model['ye_2'], model['Wd_1']), model['Bd_1']), name = 'yd-1')
	
	# SECOND DECODING LAYER

	with tf.name_scope('decoding-layer-2'):
		model['Wd_2'] = tf.Variable(tf.random_normal([dim_dec_1, dim_dec_2], stddev=1.0/dim_dec_2), name = 'Wd-2')
		model['Bd_2'] = tf.Variable(tf.random_normal([1, dim_dec_2], stddev=1.0/dim_dec_2), name = 'Bd-2')

	with tf.name_scope('output'):
		model['op'] = tf.nn.relu(tf.add(tf.matmul(model['yd_1'], model['Wd_2']), model['Bd_2']), name = 'output-vector')
		#model['op'] = tf.Print(model['op'], [model['op']], message="This is op: ")

	with tf.name_scope('loss_optim_2'):

		#model['cost'] = tf.reduce_mean(tf.pow(model['ip'] - model['op'], 2), name='cost')
		model['cost'] = tf.reduce_mean(tf.squared_difference(model['ip'], model['op']), name = 'cost')


		model['optimizer'] = tf.train.AdamOptimizer(lrate).minimize(model['cost'], name = 'optim')
		model['sum_loss'] = tf.summary.scalar(model['cost'].name, model['cost'])
		model['print-cost'] = tf.Print(model['cost'], [model['cost']], message="This is cost: ")

	# return the model dictionary

	return model;
	'''
	with tf.Session() as sess:

		new_saver = tf.train.import_meta_graph('model.meta')
		new_saver.restore(sess, tf.train.latest_checkpoint('./'))
		in_vector = X_test[0:32]
		print(sess.run(model['op'], feed_dict={model['ip']: in_vector}))


		'''

def train_model(model, X_train, epoch = 100):

	with tf.Session() as session:
		init = tf.global_variables_initializer()
		session.run(init)

		path_model = './model-2'
		path_logdir = 'logs-auto-2'

		saver = tf.train.Saver()
		writer = tf.summary.FileWriter(path_logdir, session.graph)

		for i in range(epoch):
			for count in range(0,X_train.shape[0] - 31,32):
				in_vector = X_train[count:count+32]
				#in_vector = in_vector.reshape(1, in_vector.shape[0])
				feed = {model['ip']: in_vector}

				_, summary = session.run([model['optimizer'], model['sum_loss']], feed_dict = feed)
				#print('Cost: ', model['cost'].eval())
			writer.add_summary(summary, i)

			print('Epoch: ', i)

		saver.save(session, path_model)

		print('Parameters: ')
		print('We 1 \n ------------------------')
		print(model['We_1'].eval())
		print('Be 1 \n ------------------------')
		print(model['Be_1'].eval())
		print('We 2 \n ------------------------')
		print(model['We_2'].eval())
		print('Be 2 \n ------------------------')
		print(model['Be_2'].eval())
		print('Wd 1 \n ------------------------')
		print(model['Wd_1'].eval())
		print('Bd 1 \n ------------------------')
		print(model['Bd_1'].eval())
		print('Wd 2 \n ------------------------')
		print(model['Wd_2'].eval())
		print('Bd 2 \n ------------------------')
		print(model['Bd_2'].eval())
		

		'''
		## TESTING
		#print('Test ------------------------------------------------------------')
		for count in range(X_test.shape[0]):
			print(Y_test[count],' -- ')
			in_vector2 = X_test[count]
			in_vector2 = in_vector2.reshape(1, in_vector2.shape[0])
			feed2 = {model['ip']: in_vector2}
			_, summary = session.run([model['cost'], sum_loss], feed_dict = feed2)
			#writer.add_summary(summary, i)

			print(model['cost'].eval())
		
		'''

def test_model(model, X_test, Y_test):

	path_model = './model-2'
	#path_logdir = 'logs-auto-2'

	with tf.Session() as sess:

		saver = tf.train.Saver()
		saver.restore(sess, path_model)

		for i in range(0,X_test.shape[0],32):

			in_vector = X_test[i:i+32]
			feed = {model['ip']: in_vector}

			sess.run(model['print-cost'], feed_dict =  feed)
			print('---',1 in Y_test[i:i+32])




def main():
	
	X_test, X_train, Y_test = make_data()

	model = create_model(X_test, X_train, Y_test)
	train_model(model, X_train)
	#test_model(model, X_test, Y_test)
    
main()
