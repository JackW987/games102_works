import numpy as np
from matplotlib import pyplot as plt
import torch
import math
import torch.nn as nn
# data
t = torch.linspace(-math.pi,math.pi,100)
a = 3
x = a * (2*torch.cos(t)-torch.cos(2*t))
y = a * (2*torch.sin(t)-torch.sin(2*t))

#net
class love(nn.Module):
    def __init__(self,input,output):
        super(love,self).__init__()
        self.input =nn.Linear(input,20,bias = True) 
        self.hidden = nn.Linear(20,20,bias = True)
        self.out = nn.Linear(20,output,bias = True)
    def forward(self,x):
        out_1 = torch.sigmoid(self.input(x))
        out_2 = torch.sigmoid(self.hidden(out_1))
        out_3 = self.out(out_2)
        return out_3
net = love(100,100)
#loss
loss_func = torch.nn.MSELoss()
#optimiter
optimiter = torch.optim.SGD(net.parameters(),lr = 0.1)
#train
for i in range(1000):
    pred = net.forward(x)
    loss = loss_func(pred,y)

    optimiter.zero_grad()
    loss.backward()
    optimiter.step()
    print('pred:{},loss:{}'.format(pred,loss))
#test
plt.plot(x,pred.detach().numpy(),'r-')
plt.plot(x,y,'x')
plt.show()
