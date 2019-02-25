import sys
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from select_feature import return_k_best



from scipy.spatial import ConvexHull
import hdbscan


def make_data():
	RANDOM_SEED = 42
	df = pd.read_csv('creditcard.csv')
	data = df
	#data = data.sample(frac=1).reset_index(drop=True)
	data['Time'] = data['Time'].apply(lambda x : ((x//60)%1440)/1440) # convert to time of day in minuites
	cols = data.columns.values
	for col in cols[1:len(cols)]:
		data[col] = MinMaxScaler().fit_transform(data[col].values.reshape(-1, 1))	
	data['Class'] = data['Class'].apply(lambda x : int(x//1))

	data = return_k_best(data, k_features=12)
	# Prepare for train
	Y = data['Class']
	X = data.drop(['Class'], axis=1).values

	return X,Y
'''
def order(vec1, vec2):
	assert len(vec1) == len(vec2)
	e=1
	for i in range(len(vec1)):
		e = e and (vec1[i]==vec2[i])
'''

def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.euclidean):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 

    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        if j%100 == 0:
        	print(j)
        for i, x_core in enumerate(dbscan_model.components_): 
            if metric(x_new, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break

    return y_new


def main():

	verbose=1
	X, Y = make_data()
	dbscan = DBSCAN(eps=0.2, min_samples=2, n_jobs=4)
	T=X[0:32768]
	#print(T)
	T2=X[32768:39000]
	pred = np.array(dbscan.fit_predict(T))
	#clus = hdbscan.HDBSCAN(min_cluster_size=2, cluster_selection_method='leaf',algorithm='prims_kdtree')

	#pred = dbscan.labels_
	#pred = np.array(pred)
	#pred = np.array(clus.fit_predict(T))
	arr = np.insert(T,len(T[0]), pred, axis=1)
	
	np.savetxt("dbscan.txt", arr, fmt="%f")
	#TT=np.loadtxt("dbscan.txt")
	#T=np.array(TT[:,:30])
	print (T.shape)
	print(dbscan.components_.shape)
	#pred =np.array(TT[:,30:31])
	print (pred.shape)
	print(np.amax(pred))
	print(np.amin(pred))
	print(pred)

	classes = dict()
	for i in range(len(T)):
		p=int(pred[i])
		if p not in classes: classes[p] = 0
		classes[p]=classes[p]+1

	for x in classes: 
		print('class', x, classes[x])
	
	tp = tn = fp = fn =0
	tp1 = tn1 = fp1 = fn1 =0
	for i in range(len(T)):
		if Y[i]==0:
			if pred[i]==0:
				tp+=1
				fn1+=1
			else:
				fn+=1
				tp1+=1
		if Y[i]==1:
			if pred[i] != 0:
				tn+=1
				fp1+=1
			else:
				fp+=1
				tn1+=1

	print(tp, ' --- ', tn, ' --- ', fp, ' --- ', fn)
	print(tp1, ' --- ', tn1, ' --- ', fp1, ' --- ', fn1)
	
	y_new = dbscan_predict(dbscan, T2)

	tp = tn = fp = fn =0
	tp1 = tn1 = fp1 = fn1 =0
	for i in range(len(T2)):
		if Y[i+32768]==0:
			if y_new[i]==0:
				tp+=1
				fn1+=1
			else:
				fn+=1
				tp1+=1
		if Y[i+32768]==1:
			if y_new[i] != 0:
				tn+=1
				fp1+=1
			else:
				fp+=1
				tn1+=1

	print(tp, ' --- ', tn, ' --- ', fp, ' --- ', fn)
	print(tp1, ' --- ', tn1, ' --- ', fp1, ' --- ', fn1)
	

main()


