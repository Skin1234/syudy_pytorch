import torch
from torch import autograd
import numpy as np

x = torch.tensor(1.)
a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
c = torch.tensor(3., requires_grad=True)

y = a**2 * x + b * x + c

print('before:', a.grad, b.grad, c.grad)
grads = autograd.grad(y, [a, b, c])
print('after:', grads[0], grads[1], grads[2])  #abc的偏微分

torch.tensor(1.)
z = torch.randn(2,3)  #在cpuTrue
print(a.type())
print(isinstance(z,torch.cuda.FloatTensor))  #False
z = z.cuda()  #搬到GPU上
print(isinstance(z,torch.cuda.FloatTensor))  #

k = 1.  #python
torch.tensor(k)  #pytorch

p = torch.tensor(2.2)
print(p.shape)  #torch.Size([])
len(p.shape)  #0
print(p.size())  #torch.Size([])

print(torch.tensor([1.1]))
print(torch.tensor([1.1,2.2]))
print(torch.FloatTensor(1))  #数据初始化 ，参数：向量长度 tensor([0.])
data = np.ones(2)  #[1. 1.]
print(data.shape)  #(2,)
print(data.size()+"+++++")
print(torch.from_numpy(data))  #tensor([1., 1.], dtype=torch.float64)

q = torch.rand(1,2,3)
print(q)
print(q.shape)
print(q[0])
print(q.numel())  #占用内存6
print(q.dim())   #3


#numpy-pytorch类型转换
h = np.array([2,3.3])
print(torch.from_numpy(a))  #tensor([2.0000, 3.3000], dtype=torch.float64)
h = np.ones([2,3])
print(torch.from_numpy(a))  #tensor([[1., 1., 1.],[1., 1., 1.]], dtype=tor4ch.float64)