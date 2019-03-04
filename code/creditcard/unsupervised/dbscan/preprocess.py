import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(filename):
	df = pd.read_csv(filename)
	data = df
	data['Time'] = data['Time'].apply(lambda x : ((x//60)%1440)/1440) # convert to time of day in minuites
	cols = data.columns.values
	for col in cols[1:len(cols)]:
		data[col] = MinMaxScaler().fit_transform(data[col].values.reshape(-1, 1))	
	data['Class'] = data['Class'].apply(lambda x : int(x//1))

	data = data.drop(['Time'], axis=1)
	# If feature selection
	#data = return_k_best(data, k_features=12)

	# Prepare for train
	Y = data['Class'].values
	X = data.drop(['Class'], axis=1).values

	return X,Y
