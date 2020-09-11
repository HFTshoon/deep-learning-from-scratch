import sys,os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net_prac import TwoLayerNet

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)

iter_count=10000
batch_size=100
epoch_size=x_train.shape[0]/batch_size

network=TwoLayerNet(784,50,10)
learning_rate=0.1

for i in range(iter_count):
	batch_mask=np.random.choice(x_train.shape[0],batch_size)
	x_batch=x_train[batch_mask]
	t_batch=t_train[batch_mask]

	grad=network.gradient(x_batch,t_batch)
	for key in ['W1','b1','W2','b2']:
		network.params[key]-=learning_rate*grad[key]

	if i%epoch_size==0:
		acc1=network.acc(x_train,t_train)
		acc2=network.acc(x_test,t_test)
		print(acc1,acc2)

