import linear_regres_methods as lrm
import os

def main():


	path = os.getcwd() + '/sampledata1.txt'
	#setup data object where we can access all the relevant parameters for calculations
	data_ex = lrm.Data(path)
		
	Theta, Cost = lrm.gradient_descent_algo()

	lrm.Create_graph()


	
main()

