import os,sys
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
from neuralnet_mnist import get_data,init_network,predict

import pickle
import numpy as np

if __name__ == "__main__":
	x,t=get_data()
	network=init_network()

	batch_size=100
	acc=0

	for i in range(0,len(x),batch_size):
		x_batch=x[i:i+batch_size]
		y_batch=predict(network,x_batch)
		p=np.argmax(y_batch,axis=1)
		acc+=np.sum(p==t[i:i+batch_size])
	print(float(acc)/len(x))