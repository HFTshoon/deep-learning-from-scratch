import numpy as np
from gradient_2d import numerical_gradient

def gradient_descent(f,init_x,lr=0.01,step_num=100):
	x=init_x

	for i in range(step_num):
		grad=numerical_gradient(f,x)
		x-=lr*grad
	return x

def fun2(x):
	return x[0]**2+x[1]**2

if __name__ == "__main__":
	init_x=np.array([-3.0,4.0])
	print(gradient_descent(fun2,init_x=init_x,lr=0.1,step_num=100))
