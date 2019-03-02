import sys
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from select_feature import return_k_best
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from scipy.spatial import ConvexHull
from collections import Counter

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

	data = data.drop(['Time'], axis=1)
	#data = return_k_best(data, k_features=12)
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
'''
def dbscan_predict(X_pred, kdtree, pred):

	Y_pred = list()
	#indices = kdtree.query_radius(X_pred, r=0.23)
	indices = kdtree.query_radius(X_pred, r=0.2)
	for arr in indices:
		if len(arr) == 0: Y_pred.append(-1)
		else:
			clus = [pred[i] for i in arr]
			clus_ctr = Counter(clus)
			label = clus_ctr.most_common(1)[0][0]
			#print ('Cluster: ', clus)
			#print('Label: ',label)
			Y_pred.append(label)

	return Y_pred



def main():

	verbose=1
	X, Y = make_data()
	T=X[0:32768]
	T2=X[40000:45000]
	print('Preparing KD tree')
	kdtree = KDTree(T)
	print('Computing radius neighbour graph ...')
	neighbors_model = NearestNeighbors(radius=0.2, n_neighbors=3, n_jobs=4)
	neighbors_model.fit(kdtree)
	rn_graph = neighbors_model.radius_neighbors_graph(T, 0.2, mode='distance')
	print('Radius neighbour graph computed')
	
	print('Applying DBSCAN ...')
	#print(T)
	T=X[0:32768]
	dbscan = DBSCAN(eps=0.2, min_samples=2, metric='precomputed', algorithm="ball_tree", n_jobs=4)
	pred = np.array(dbscan.fit_predict(rn_graph))
	print('DBSCAN done')
	arr = np.insert(T,len(T[0]), pred, axis=1)
	
	np.savetxt("dbscan2.txt", arr, fmt="%f")
	#TT=np.loadtxt("dbscan2.txt")
	#T=np.array(TT[:,:30])
	print (T.shape)
	#print(dbscan.components_.shape)
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
		else:
			if pred[i] != 0:
				tn+=1
				fp1+=1
			else:
				fp+=1
				tn1+=1

	print(tp, ' --- ', tn, ' --- ', fp, ' --- ', fn)
	print(tp1, ' --- ', tn1, ' --- ', fp1, ' --- ', fn1)
	
	y_new = dbscan_predict(T2, kdtree, pred)

	an = Counter(y_new)
	print(an)
	tp = tn = fp = fn =0
	tp1 = tn1 = fp1 = fn1 =0
	for i in range(len(T2)):
		if Y[i]==0:
			if y_new[i]==0:
				tp+=1
				fn1+=1
			else:
				fn+=1
				tp1+=1
		else:
			if y_new[i] != 0:
				tn+=1
				fp1+=1
			else:
				fp+=1
				tn1+=1

	print(tp, ' --- ', tn, ' --- ', fp, ' --- ', fn)
	print(tp1, ' --- ', tn1, ' --- ', fp1, ' --- ', fn1)
	

main()


