import torch
import torch.nn.functional as F
#Sigmoid
a = torch.linspace(-100,100,10)
S = torch.nn.Sigmoid()
print(S(a))

#Tanh
b = torch.linspace(-1,1,10)
T = torch.nn.Tanh()
print(T(b))

#Relu
R = torch.nn.ReLU()
print(R(b))

#loss  https://www.bilibili.com/video/BV1cV411Y7jZ?p=30
x = torch.ones(1)  #predict


w = torch.full([1], 2.)  #target

mse = torch.nn.MSELoss()  #求predict和target之间的loss。
mse_result = mse(torch.ones(1), x*w)
print(mse_result)  #tensor(1.)

#w = torch.tensor([1.],requires_grad=True)   #创建时设置w可更新

w.requires_grad_()  #设置w是需要更新的变量
mse_result = mse(torch.ones(1), x*w)
#方法1：
#print(torch.autograd.grad(mse_result,[w]))  #(tensor([2.]),)loss对w的偏导,返回一个list
#方法2：
mse_result.backward(retain_graph=True)
print(w.grad)  #tensor([2.]) loss对所有图结构的w求导，返回在每一个W1/w2。。。变量上，如：w1.grad

#softmax  拉大高分与低分的距离 https://www.bilibili.com/video/BV1cV411Y7jZ?p=31
c = torch.rand(3)
c.requires_grad_()
p = F.softmax(c,dim=0)
print(p)
p.backward(torch.ones_like(c))  #同试图第二次向后遍历图形，但是保存的中间结果已经被释放了。当第一次向后调用时指定retain_graph=True，图不被清除。
print(c.grad)                                                            #(torch.ones_like(c)输出和c同样大小的shape

p = F.softmax(c,dim=0)
print(torch.autograd.grad(p[1],[c],retain_graph=True))  #(tensor([-0.0801,  0.2092, -0.1290]),)  ij相同位置为+不同位置为-
print(torch.autograd.grad(p[2],[c]))  #(tensor([-0.1165, -0.1290,  0.2455]),)