class Relu:
	def __init__(self):
		self.mask=None

	def forward(self,x):
		self.mask=x<0
		result=x.copy()
		out[self.mask]=0
		return out

	def backward(self,incoming):
		incoming[self.mask]=0
		result=incoming
		return result 