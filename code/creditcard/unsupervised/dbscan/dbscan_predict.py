import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from collections import Counter

class DBSCAN_Predict:

	def __init__(self, eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', n_jobs=None):

		self.eps = eps
		self.min_samples = min_samples
		self.metric = metric
		self.algorithm = algorithm
		self.n_jobs = n_jobs
		self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm, n_jobs=n_jobs)

	def dbscan_predict_(self, X_predict):

		Y_pred = list()
		indices = self.kdtree.query_radius(X_predict, r=self.eps)
		for arr in indices:
			if len(arr) == 0: Y_pred.append(-1) # outlier
			else:
				clus = [self.core_labels[i] for i in arr]
				clus_ctr = Counter(clus)
				label = clus_ctr.most_common(1)[0][0] # find the label of majority of points in eps neighbourhood
				Y_pred.append(label)

		return Y_pred

	def make_tree_(self, data):

		core_indices = self.dbscan_model.core_sample_indices_ # Indices of core samples		
		core_samples = [data[i] for i in core_indices] # Copy of core samples
		self.core_labels = [self.class_labels[i] for i in core_indices] # labels of the core indices
		print('Preparing KD tree')
		self.kdtree = KDTree(core_samples)

	def fit(self, X, y=None):

		self.dbscan_model.fit(X,y)
		self.class_labels = self.dbscan_model.labels_
		self.make_tree_(X)

	def fit_predict(self, X, y=None):

		Y = self.dbscan_model.fit_predict(X,y)
		self.class_labels = self.dbscan_model.labels_
		self.make_tree_(X)
		return Y

	def predict(self, X_predict):

		y_new = self.dbscan_predict_(X_predict)
		return y_new


	def get_core_points(self):

		return self.dbscan_model.components_

	def get_labels(self):

		return self.class_labels

	def get_core_indices(self):

		return self.dbscan_model.core_sample_indices_
