import numpy as np

def relu(x):
	return np.maximum(0,x)

"""
import matplotlib.pyplot as plt
from step_function import step_function
from sigmoid import sigmoid

x=np.arange(-5.0,5.0,0.1)
plt.plot(x,step_function(x),x,sigmoid(x),x,relu(x))
plt.ylim(-0.1,1.1)
plt.show()
"""
