import sys,os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from common.optimizer import *
from ch05.two_layer_net_prac import TwoLayerNet
import matplotlib.pyplot as plt
from common.util import smooth_curve

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)

keys=['SGD','Momentum','AdaGrad','Adam']

optimizers={}
optimizers['SGD']=SGD()
optimizers['Momentum']=Momentum()
optimizers['AdaGrad']=AdaGrad()
optimizers['Adam']=Adam()

networks={}
losses={}
for key in keys:
	networks[key]=TwoLayerNet(784,50,10)
	losses[key]=[]


iter_count=10000
batch_size=100
train_size=x_train.shape[0]
epoch_size=train_size/batch_size

for i in range(iter_count):
	batch_mask=np.random.choice(train_size,batch_size)
	x_batch=x_train[batch_mask]
	t_batch=t_train[batch_mask]

	grads={}
	for key in keys:
		optimizers[key].update(networks[key].params,networks[key].gradient(x_batch,t_batch))
		losses[key].append(networks[key].loss(x_batch,t_batch))

	if i%epoch_size==0:
		print("====================iter_count : "+str(i))
		for key in keys:
			accloss=networks[key].acc(x_test,t_test)
			print(str(key)+" : "+str(accloss))
			
for key in keys:
	plt.plot(range(iter_count),smooth_curve(losses[key]),markevery=100)
plt.ylim(0,1)
plt.show()