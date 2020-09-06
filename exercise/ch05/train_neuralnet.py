import sys,os
sys.path.append(os.pardir)

import numpy as np
from two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)
network=TwoLayerNet(784,50,10)

iters_num=10000
train_size=x_train.shape[0]
batch_size=100
epoch_size=train_size/batch_size
learning_rate=0.1

train_loss_list=[]
train_acc_list=[]
test_acc_list=[]

for i in range(iters_num):
	batch_mask=np.random.choice(train_size,batch_size)
	x_batch=x_train[batch_mask]
	t_batch=t_train[batch_mask]

	grad=network.gradient(x_batch,t_batch)

	for key in ('W1','b1','W2','b2'):
		network.params[key]-=grad[key]*learning_rate

	loss=network.loss(x_batch,t_batch)
	train_loss_list.append(loss)

	if i%epoch_size==0:
		train_acc=network.acc(x_train,t_train)
		test_acc=network.acc(x_test,t_test)
		train_acc_list.append(train_acc)
		test_acc_list.append(test_acc)
		print(train_acc,test_acc)