import linear_regres_methods as lrm
import unittest

class test_linear_regression(unittest.TestCase):
	
	def test_check_shape_matrix(self):
		self.assertEqual(lrm.columnofmatrix([[1,2],[3,4]]),2)
		self.assertEqual(lrm.rowofmatrix([[1,2],[3,4]]),2)

	def test_cost_descreasing(self):
		self.assertTrue(lrm.iscostdecreasing([5,4,3,2,1]))


if __name__ == '__main__':
	unittest.main()
