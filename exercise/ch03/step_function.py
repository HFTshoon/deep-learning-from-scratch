"""
def step_function(x):
	return 1 if x>0 else 0
"""
# 이경우 인수를 하나만 받을 수 있고 numpy에서 쓸 행렬은 불가능

import numpy as np

def step_function(x):
	y=x>0
	return y.astype(np.int)