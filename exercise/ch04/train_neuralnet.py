#수치미분은 너무 느려서 안 돌아감

import os,sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)

train_loss_list=[]

iters_num=10000
train_size=x_train.shape[0]
batch_size=100
learning_rate=0.1

network=TwoLayerNet(784,50,10)

for i in range(iters_num):
	batch_mask=np.random.choice(train_size,batch_size)
	x_batch=x_train[batch_mask]
	t_batch=t_train[batch_mask]

	grad = network.numerical_gradient(x_batch, t_batch)
    
	for key in ('W1','b1','W2','b2'):
		network.params[key]-=grad[key]*learning_rate

	loss = network.loss(x_batch,t_batch)
	train_loss_list.append(loss)
	if i%1000==0:
		print(loss)

plt.plot(range(len(train_loss_list)),train_loss_list)
plt.ylim(0,10)
plt.show()