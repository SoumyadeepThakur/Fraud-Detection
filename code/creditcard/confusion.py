import sys
import os
import numpy as np

def read_res(filename):
	i=0
	costs = np.array([],  dtype=np.float32)
	with open(filename, 'r+') as file:
		for line in file:
			i+=1
			if i%2 == 0:
				temp = str(line[line.find("[")+1:line.find("]")])
				arr = np.fromstring(temp, dtype=np.float32, sep = ', ')
				costs = np.append(costs, arr)
	
	return costs

def evaluate(labels, file, thresh = 3):

	tp=0
	tn=0
	fp=0
	fn=0
	costs = read_res(file)
	for i in range(len(costs)):
		if labels[i] == 1:
			if costs[i]>thresh: tn+=1
			else: fp+=1

		else:
			if costs[i]>thresh: 
				fn+=1
			else: tp+=1			

	return (tp, tn,  fp, fn)
#main()