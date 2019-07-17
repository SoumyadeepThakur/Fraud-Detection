import tensorflow as tf

def autoencoder_model(input_shape, lrate = 1e-5):

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


def contractive_autoencoder_model(input_shape, lrate = 1e-5):

	#initialize data
	#X_test, X_test, Y_test = make_data()

	#initialize parameters
	model = dict()

	dim_b = 32
	dim_input = input_shape
	dim_enc_1 = 25
	dim_hidden = 19
	dim_dec_1 = 25
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
		model['We_2'] = tf.Variable(tf.random_normal([dim_enc_1, dim_hidden], stddev=1.0/dim_hidden), name = 'We-2')
		model['Be_2'] = tf.Variable(tf.random_normal([1, dim_hidden], stddev=1.0/dim_hidden), name = 'Be-2')
		model['ye_2'] = tf.nn.sigmoid(tf.add(tf.matmul(model['ye_1'], model['We_2']), model['Be_2']), name = 'ye-2')
		


	
	# FIRST DECODING LAYER

	with tf.name_scope('decoding-layer-1'):
		model['Wd_1'] = tf.Variable(tf.random_normal([dim_hidden, dim_dec_1], stddev=1.0/dim_dec_1), name = 'Wd-1')
		model['Bd_1'] = tf.Variable(tf.random_normal([1, dim_dec_1], stddev=1.0/dim_dec_1), name = 'Bd-1')
		model['yd_1'] = tf.nn.tanh(tf.add(tf.matmul(model['ye_2'], model['Wd_1']), model['Bd_1']), name = 'yd-1')

	# SECOND DECODING LAYER

	with tf.name_scope('decoding-layer-2'):
		model['Wd_2'] = tf.Variable(tf.random_normal([dim_dec_1, dim_dec_2], stddev=1.0/dim_dec_2), name = 'Wd-2')
		model['Bd_2'] = tf.Variable(tf.random_normal([1, dim_dec_2], stddev=1.0/dim_dec_2), name = 'Bd-2')

	

	with tf.name_scope('output'):
		model['op'] = tf.nn.sigmoid(tf.add(tf.matmul(model['yd_1'], model['Wd_2']), model['Bd_2']), name = 'output-vector')
		#model['op'] = tf.Print(model['op'], [model['op']], message="This is op: ", summarize=960)


	with tf.name_scope('loss_optim'):

		#model['cost'] = tf.reduce_mean(tf.pow(model['ip'] - model['op'], 2), name='cost')
		model['cost-regen'] = tf.reduce_mean(tf.squared_difference(model['ip'], model['op']), name = 'regen-cost')
		model['contr-1'] = tf.stack([tf.gradients(model['op'][:, i], model['ye_2']) for i in range(dim_input)], axis=2)
		model['jac'] = tf.reduce_mean(model['contr-1'], axis=1)
		model['frob-jac'] = tf.norm(model['jac'], ord='fro', axis=[1,2])
		#model['Jac'] = jacobian(model['op'], model['ip'])
		#model['contr-loss'] = tf.norm(model['Jac'], ord='fro', axis=(0,1), name='contr-loss')
		model['frob-jac'] = tf.multiply(model['frob-jac'], model['cost-regen'])
		model['cost'] = tf.reduce_mean(tf.add(model['cost-regen'], model['frob-jac']), name='cost')
		model['cost-2'] = tf.reduce_mean(tf.squared_difference(model['ip'], model['op']), axis=1, name='cost-2')

		model['optimizer'] = tf.train.AdamOptimizer(lrate).minimize(model['cost'], name = 'optim')
		model['sum_loss'] = tf.summary.scalar(model['cost'].name, model['cost'])
		model['print-cost'] = tf.Print(model['cost-2'], [model['cost-2']], message="This is cost: ", summarize = 32)

	# return the model dictionary

	return model;