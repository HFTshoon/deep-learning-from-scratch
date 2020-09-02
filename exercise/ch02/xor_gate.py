import numpy as np

tflist=[[0,0],[0,1],[1,0],[1,1]]
def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    tmp=np.sum(x*w)+b
    return 0 if tmp<0 else 1

def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.2
    tmp=np.sum(w*x)+b
    return 0 if tmp<0 else 1

def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    tmp=np.sum(x*w)+b
    return 0 if tmp<0 else 1

def XOR(x1,x2):
    x=NAND(x1,x2)
    y=OR(x1,x2)
    return AND(x,y)

for ls in tflist:
    print("AND"+str(ls)+"---"+str(AND(ls[0],ls[1])))
    print("OR"+str(ls)+"---"+str(OR(ls[0],ls[1])))
    print("NAND"+str(ls)+"---"+str(NAND(ls[0],ls[1])))
    print("XOR"+str(ls)+"---"+str(XOR(ls[0],ls[1])))