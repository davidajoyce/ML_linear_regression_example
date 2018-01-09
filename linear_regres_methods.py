import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

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
	
	if not check_cost_decrease(cost):
		sys.stderr.write('error: cost is not decreasing, check gradient descent algorithm')
		

	return theta, cost

def compute_the_cost(X,Y,theta):

	inner =np.power(((X*theta.T)-Y),2)
	
	#least square function linear regression method
	cost = np.sum(inner)/(2*len(X))

	return cost

def check_cost_decrease(cost):
	size = len(cost)
	
	for i in range(size-1):
		
		if cost[i] < cost[i+1]:
			return False

	return True	
	

def create_graph(data, theta, iters, cost):


	graph = data.plot(kind='scatter', x='Population', y='Profit',figsize=(12,8))

	#first figure see what the scatter plot looks like
	plt.show()

	x = np.linspace(data.Population.min(), data.Population.max(), 100)  
	f = theta[0, 0] + (theta[0, 1] * x)

	fig, ax = plt.subplots(figsize=(12,8))  
	ax.plot(x, f, 'r', label='Prediction')  
	ax.scatter(data.Population, data.Profit, label='Traning Data')  
	ax.legend(loc=2)  
	ax.set_xlabel('Population')  
	ax.set_ylabel('Profit')  
	ax.set_title('Predicted Profit vs. Population Size')  

	#display the prediction based on the training data scatter plot above
	plt.show()

	#display how the error and cost changes throughout the iterations
	fig, ax = plt.subplots(figsize=(12,8))  
	ax.plot(np.arange(iters), cost, 'r')  
	ax.set_xlabel('Iterations')  
	ax.set_ylabel('Cost')  
	ax.set_title('Error vs. Training Epoch')  

	plt.show()



	
