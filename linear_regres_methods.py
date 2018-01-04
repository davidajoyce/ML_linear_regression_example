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

	#add a column of 1's to accomodate with the intercept term
	data.insert(0, 'Ones', 1)

	return data

	
def gradient_descent_algo(X, Y, theta, alpha, iters):

	temp = np.matrix(np.zeros(theta.shape))

	#size of theta matrix i.e parameters = 2
	parameters = int(theta.ravel().shape[1])

	cost = np.zeros(iters)
	
	for i in range(iters):
		#calculate the error of predicted value y(theta)
		error = (X*theta.T)-Y
		
		for j in range(parameters):
			term = np.multiply(error, X[:,j])
			temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
	
		theta = temp
		cost[i] = compute_the_cost(X,Y,theta)
		#print "costing array"
		#print cost[i]


	return theta, cost

def compute_the_cost(X,Y,theta):

	inner =np.power(((X*theta.T)-Y),2)
	
	#least square function linear regression method
	cost = np.sum(inner)/(2*len(X))

	return cost

	
	

#def Create_graph():

	
