import linear_regres_methods as lrm
import os
import numpy as np

def main():

	iters = 1000
	alpha = 0.01

	path = os.getcwd() + '/sampledata1.txt'
	#setup data object where we can access all the relevant parameters for calculations
	
	#create table from txt file 
	data_table = lrm.create_data_table(path)

	data_matrix = lrm.Data(data_table)

	theta = np.matrix(np.array([0,0]))

	#print "test class"
	#print data_matrix.X	
		
	Theta, Cost = lrm.gradient_descent_algo(data_matrix.X, data_matrix.Y, theta, alpha, iters)


	#lrm.Create_graph()


	
main()
