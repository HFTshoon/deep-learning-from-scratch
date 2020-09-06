import sys,os
sys.path.append(os.pardir)

import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
	def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
		self.params={}
		self.params['W1']=weight_init_std*np.random.randn(input_size,hidden_size)
		self.params['b1']=np.zeros(hidden_size)
		self.params['W2']=weight_init_std*np.random.randn(hidden_size,output_size)
		self.params['b2']=np.zeros(output_size)

		self.layers=OrderedDict()
		self.layers['Af1']=Affine(self.params['W1'],self.params['b1'])
		self.layers['Rl1']=Relu()
		self.layers['Af2']=Affine(self.params['W2'],self.params['b2'])

		self.lastlayer=SoftmaxWithLoss()

	def predict(self,x):
		for layer in self.layers.values():
			x=layer.forward(x)
		return x

	def loss(self,x,t):
		a=self.predict(x)
		return self.lastlayer.forward(a,t)

	def acc(self,x,t):
		a=self.predict(x)
		y=softmax(a)
		y=np.argmax(y,axis=1)
		if t.ndim!=1:
			t=np.argmax(t,axis=1)

		return np.sum(y==t)/float(x.shape[0])

	def gradient(self,x,t):
		self.loss(x,t)
		
		incoming=1
		incoming=self.lastlayer.backward(incoming)

		layers=list(self.layers.values())
		layers.reverse()
		for layer in layers:
			incoming=layer.backward(incoming)

		grads={}
		grads['W1']=self.layers['Af1'].dW
		grads['b1']=self.layers['Af1'].db
		grads['W2']=self.layers['Af2'].dW
		grads['b2']=self.layers['Af2'].db

		return grads