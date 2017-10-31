import numpy as np
from scipy.spatial.distance import cdist


# kernel_trick() computes kernel function on x1 and x2.
# Currently uses only gaussian kernels.
# Written in a way so that other kernels can be incorporated
# in the future.
def kernel_trick(x1, x2, kernel_type='gaussian', const1=1):
	# Using Gaussian kernels
	if kernel_type == 'gaussian':
		kernel_matrix = \
			np.exp( cdist(x1, x2, metric='sqeuclidean') \
			/ (2 * const1**2) )
		# Deal with infinity that can arise from exp()
		kernel_matrix = np.fmin(kernel_matrix, np.finfo(np.float64).max)
		kernel_matrix = 1 / kernel_matrix
	return kernel_matrix


# Create object of this class, then run object.fit(data) to
# get the centers and memberships
# Class member variables set by the constructor are :
# k : Number of centers
# m : FCM parameter
# p : KHM parameter
# kernel : Set to 'gaussian' \
# num_iters : Number of iterations
# const1 : For gaussian kernels, it is the parameter sigma
# num_inits : Number of re-initializations
# tol : tolerance threshold
class kergfcm_gaussian:

	def __init__(self, k=1, m=2, p=2, kernel='gaussian', \
		num_iters=300, const1=1, num_inits=30, tol=1e-16):
		self.k = k
		self.m = m
		self.p = p
		self.kernel = kernel
		self.num_iters = num_iters
		self.const1 = const1
		self.num_inits = num_inits
		self.tol = tol

	def getvals(self) :
		return \
			self.k, self.m, self.p, self.kernel, \
			self.num_iters,	self.const1, self.num_inits, self.tol

	# Warning : Returns transposed matrices Kernel_matrix, U_matrix.
	# Transpose them back before use.
	def recompute_centers(self, data, centers) :
		Kernel_matrix = kernel_trick(kernel_type=self.kernel, x1=data,\
			x2=centers, const1=self.const1).T
		Ker_dist = np.fmax(1 - Kernel_matrix, np.finfo(np.float64).eps)
		U_matrix = Ker_dist**(-self.p/(2*(self.m-1)))
		U_matrix = np.fmax(U_matrix, np.finfo(np.float64).eps)
		U_matrix /= np.ones((self.k,1)).dot(np.atleast_2d(U_matrix.sum(axis=0)))

		expr_part = (U_matrix**self.m) * (Ker_dist**((self.p-2)/2) \
		* Kernel_matrix)
		expr_part = np.fmax(expr_part, np.finfo(np.float64).eps)
		new_centers = expr_part.dot(data) \
			/( np.ones( (data.shape[1],1) ).dot(\
			np.atleast_2d(expr_part.sum(axis=1)) ).T )
		return new_centers, Kernel_matrix, U_matrix

	def terminate(self, centers, old_centers) :
		if np.sum( (centers - old_centers)**2 ) < self.tol :
			return 1
		return 0

	# Runs KGFCM : returns centers and memberships
	def fit(self, data) :
		best_cost = +np.inf
		best_centers = np.zeros((self.k, data.shape[1]))
		mem = np.zeros((data.shape[0]))

		for iter1 in range(self.num_inits) :
			centers = \
				data[np.random.permutation(data.shape[0])[0:self.k],:]
			for iter2 in range(self.num_iters) :
				old_centers = np.array(centers)
				centers, Kernel_matrix, U_matrix = \
				 	self.recompute_centers(data, old_centers)
				if self.terminate(centers, old_centers) :
					break

			# Compute cost
			um = (U_matrix.T**self.m)
			cost = np.sum(um * ((2 - 2*Kernel_matrix.T)**(self.p/2)))
			if cost < best_cost :
				best_cost = cost
				best_centers = np.array(centers)
				mem = np.argmax(U_matrix, axis=0)

		return best_centers, mem

if __name__ == '__main__' :
	# DEBUG

	'''
	#Test1 : kernel_trick()
	x1 = np.array([[0,0],[0,1],[2,0],[10,15],[-10,-20]])
	x2 = np.array([[0,0],[10,15]])
	print(kernel_trick(x1, x2, kernel_type='gaussian', const1=1))
	'''

	'''
	# Test2 : getvals()
	kgfcm1 = kergfcm_gaussian(k=2, m=2, p=2, kernel='gaussian',\
		num_iters=300, const1=1, num_inits=30, tol=1e-16)
	print(kgfcm1.getvals())
	'''

	'''
	# Test3 : recompute_centers()
	data = np.array([[0,0],[0,1],[2,0],[10,15],[-10,-20]])
	centers = np.array([[0,0],[10,15]])
	kgfcm1 = kergfcm_gaussian(k=2, m=2, p=2, kernel='gaussian',\
		num_iters=300, const1=1, num_inits=30, tol=1e-16)
	new_centers, Kernel_matrix, U_matrix = \
		kgfcm1.recompute_centers(data, centers)
	print(new_centers)
	print(U_matrix)
	print(np.argmax(U_matrix, axis=0))
	'''

	'''
	# Test4 : terminate()
	kgfcm1 = kergfcm_gaussian(k=2, m=2, p=2, kernel='gaussian',\
		num_iters=300, const1=1, num_inits=30, tol=1e-16)
	x1 = np.array([[0,0],[10,15]])
	x2 = np.array([[0,0],[10,15]])
	print(kgfcm1.terminate(x1,x2))
	x3 = np.array([[0,1],[10,16]])
	print(kgfcm1.terminate(x1,x3))
	'''

	'''
	# Test5 : fit()
	kgfcm1 = kergfcm_gaussian(k=2, m=2, p=2, kernel='gaussian',\
		num_iters=300, const1=1, num_inits=30, tol=1e-16)
	x1 = np.array([[0,0],[0,1],[2,0],[10,15],[-10,-20]])
	centers, mem = kgfcm1.fit(x1)
	print(centers)
	print(mem)
	'''

	'''
	#Test6 : fit()
	kgfcm1 = kergfcm_gaussian(k=2, m=2, p=2, kernel='gaussian',\
		num_iters=300, const1=1, num_inits=30, tol=1e-16)
	x1 = np.random.rand(20,2)
	centers, mem = kgfcm1.fit(x1)
	print(centers)
	print(mem)
	'''

	'''
	#Test7 : fit()
	kgfcm1 = kergfcm_gaussian(k=2, m=2, p=2, kernel='gaussian',\
		num_iters=300, const1=1, num_inits=30, tol=1e-16)
	from sklearn.datasets import load_iris
	x1 = load_iris().data
	centers, mem = kgfcm1.fit(x1)
	import matplotlib.pyplot as plt
	for iter1 in range(2) :
		plt.scatter(x1[mem==iter1,2], x1[mem==iter1,3], marker='x')
	plt.show()
	'''

	'''
	#Test8 : fit()
	kgfcm1 = kergfcm_gaussian(k=3, m=2, p=2, kernel='gaussian',\
		num_iters=300, const1=0.72, num_inits=30, tol=1e-16)#0.72
	from sklearn.datasets import load_iris
	x1 = load_iris().data
	centers, mem = kgfcm1.fit(x1)
	from sklearn.metrics import adjusted_rand_score as ARI
	print('ARI =', ARI(load_iris().target, mem))
	import matplotlib.pyplot as plt
	for iter1 in range(3) :
		plt.scatter(x1[mem==iter1,2], x1[mem==iter1,3], marker='x')
	plt.show()
	'''

	'''
	#Test9 : fit()
	kgfcm1 = kergfcm_gaussian(k=3, m=2, p=4, kernel='gaussian',\
		num_iters=300, const1=1, num_inits=30, tol=1e-16)
	x1 = np.vstack(( np.vstack(( \
		np.random.normal(loc=[0,0], scale=1, size=(100,2)),\
		np.random.normal(loc=[6,0], scale=1, size=(100,2)) )),\
		np.random.normal(loc=[3,5], scale=1, size=(100,2)) ))
	centers, mem = kgfcm1.fit(x1)
	print(centers)
	import matplotlib.pyplot as plt
	for iter1 in range(3) :
		plt.scatter(x1[mem==iter1,0], x1[mem==iter1,1], marker='x')
	plt.show()
	'''
