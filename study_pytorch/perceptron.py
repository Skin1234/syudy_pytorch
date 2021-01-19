import torch
import torch.nn.functional as F

#单输出感知机
x = torch.randn(1,10)
w = torch.randn(1,10,requires_grad=True)
o = torch.sigmoid(x@w.t())
print(o.shape)  #torch.Size([1, 1])
loss = F.mse_loss(torch.ones(1,1),o)
print(loss)
print(loss.shape)  #torch.Size([]) 标量
loss.backward()
print(w.grad)  #tensor([[ 0.0279,  0.0686,  0.0482,  0.0179, -0.0443,  0.0192, -0.0105,  0.0458, 0.0193, -0.0539]])

#多输出感知机
x = torch.randn(1,10)
w = torch.randn(2,10,requires_grad=True)
o = torch.sigmoid(x@w.t())
loss = F.mse_loss(torch.ones(1,2),o)
loss.backward()
print(w.grad)   #tensor([[ 0.0593,  0.0677, -0.0701, -0.1338, -0.0410, -0.0533, -0.1396,  0.0976, 0.1338,  0.0266],
                #[ 0.0653,  0.0745, -0.0771, -0.1472, -0.0451, -0.0587, -0.1536,  0.1074, 0.1472,  0.0293]])