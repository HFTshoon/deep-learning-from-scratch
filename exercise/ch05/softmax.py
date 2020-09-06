import sys,os
sys.path.append(os.pardir)
from common.functions import *

class SoftmaxWithLoss:
	def __init__(self):
		self.loss=None
		self.y=None
		self.t=None

	def forward(self,x,t):
		self.t=t
		self.y=softmax(x)
		result=cross_entropy_error(self.y,self.t)
		self.loss=result
		return result

	def backward(self,incoming):
		batch_size=self.t.shape[0]
		return (self.y-self.t)/batch_size