import numpy as np

class sigmoid:
	def __init__(self):
		self.out=None

	def forward(self,x):
		result=1/(1+np.exp(-x))
		self.out=result
		return result

	def backward(self,incoming):
		return self.out*(1.0-self.out)*incoming