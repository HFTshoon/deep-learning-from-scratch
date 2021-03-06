class AdaGrad:
	def __init__(self,lr=0.01):
		self.lr=lr
		self.h=None

	def update(self,params,grads):
		if self.h is None:
			self.h={}
			for key,val in params.items():
				self.h[key]=np.zeros_like(val)

		for key in parmas.keys():
			self.h+=grads[key]*grads[key]
			params[key]-=self.lr*grads[key]/(1e-7+np.sqrt(self.h[key]))