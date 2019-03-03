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

class DBSCAN-Predict:

	def __init__(self, eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', n_jobs=None):

		self.eps = eps
		self.min_samples = min_samples
		self.metric = metric
		self.algorithm = algorithm
		self.n_jobs = n_jobs
		self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm, n_jobs=n_jobs)

	def dbscan_predict_(X_predict, kdtree):

		Y_pred = list()
		indices = kdtree.query_radius(X_predict, r=self.eps)
		for arr in indices:
			if len(arr) == 0: Y_pred.append(-1) # outlier
			else:
				clus = [pred[i] for i in arr]
				clus_ctr = Counter(clus)
				label = clus_ctr.most_common(1)[0][0] # find the label of majority of points in eps neighbourhood
				Y_pred.append(label)

		return Y_pred

	def fit(X, y=None):

		self.data 
		self.dbscan_model.fit(X,y)
		self.class_labels = self.dbscan_model.labels_

	def fit_predict(X, y=None)

		Y = self.dbscan_model.fit_predict(X,y)
		self.class_labels = self.dbscan_model.labels_
		return Y

	def predict(X_predict):

		core_indices = self.dbscan_model.core_sample_indices_ # Indices of core samples
		core_samples = self.dbscan_model.components_ # Copy of core samples
		core_labels = [self.class_labels[i] for i in core_indices] # labels of the core indices
		print('Preparing KD tree')
		self.kdtree = KDTree(core_samples)
		y_new = dbscan_predict_(X_predict)
		return y_new


	def get_core_points():

		return self.dbscan_model.components_

	def get_labels():

		return self.class_labels

	def get_core_indices():

		return self.dbscan_model.core_sample_indices_
