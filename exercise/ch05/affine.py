import numpy as np

class Affine:
	def __init__(self,W,b):
		self.W=W
		self.b=b
		self.x=None
		self.dW=None
		self.db=None

	def forward(self,x):
		self.x=x
		result=np.dot(x,self.W)+b
		return result

	def backward(self,incoming)
		result=np.dot(incoming,self.W.T)
		self.dW=np.dot(self.x.T,incoming)
		self.db=np.sum(incoming,axis=0)
		return result