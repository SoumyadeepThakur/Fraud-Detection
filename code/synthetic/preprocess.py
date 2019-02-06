import numpy as np
import signal
import sys
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#from confusion import evaluate

max_val = dict()
min_val = dict()
type = dict()
type['CASH_IN']=0.1
type['CASH_OUT']=0.3
type['DEBIT']=0.5
type['PAYMENT']=0.7
type['TRANSFER']=0.9

def make_data():

	max=0
	min=pow(2,64)
	i=0
	for chunk in pd.read_csv('synthetic.csv', chunksize=262144):
		df = chunk
		print('Read chunk ',i)
		i+=1
		df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1) # ignore flags
		cols = df.columns.values

		for col in cols[2:len(cols)-1]:
			c_max=df[col].max()
			c_min=df[col].min()
			max_val[col] = c_max if (col not in max_val or c_max > max_val[col]) else max_val[col]
			min_val[col] = c_min if (col not in min_val or c_min < min_val[col]) else min_val[col]

	print ('Normalize: ')

	i=0
	for chunk in pd.read_csv('synthetic.csv', chunksize=262144):

		df = chunk
		print('Read chunk ',i)
		df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1) # ignore flags
		df['step'] = df['step'].apply(lambda x: x%24) # scale to hour of the day
		cols = df.columns.values
		df['step'] = df['step'].apply(lambda x: (x%24)/24) # scale to hour of the day

		for col in cols[2:len(cols)-1]:
			df[col] = df[col].apply(lambda x: (x-min_val[col])/(max_val[col]-min_val[col]))

		df['type'] = df['type'].apply(lambda x: type[x]) # is this required ?

		print('Write chunk ',i)
		df.to_csv('synthetic-modified.csv', mode='a', index=False)
		i+=1

make_data()



