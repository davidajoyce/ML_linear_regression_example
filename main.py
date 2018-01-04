import linear_regres_methods as lrm
import os

def main():


	path = os.getcwd() + '/sampledata1.txt'
	#setup data object where we can access all the relevant parameters for calculations

	
	#create table from txt file 
	data_table = lrm.create_data_table(path)

	data1 = lrm.Data(data_table)

	print "test class"
	print data1.X

	#data_matrix = lrm.Data(data_X, data_Y)	
		
	#Theta, Cost = lrm.gradient_descent_algo()

	#lrm.Create_graph()


	
main()

