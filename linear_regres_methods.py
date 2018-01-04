import numpy as np
import pandas as pd


class Data(object):
	
	def __init__(self, data):
	
		self.cols = data.shape[1]
		self.x = data.iloc[:,0:self.cols-1]
		self.y = data.iloc[:,self.cols-1:self.cols]
		self.X = np.matrix(self.x.values)
		self.Y = np.matrix(self.y.values)


def create_data_table(path):
		
	data = pd.read_csv(path, header=None, names=['Population','Profit'])

	data.insert(0, 'Ones', 1)

	return data

	
#def gradient_descent_algo():
	

#def Create_graph():

	
