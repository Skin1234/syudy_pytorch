import  torch

#+-*/基本运算
a = torch.rand(3,4)
b = torch.rand(4)
print((a+b).shape) #torch.Size([3, 4])
torch.add(a,b)
print(torch.all(torch.eq(a-b, torch.sub(a,b))))
print(torch.all(torch.eq(a*b, torch.mul(a,b))))
print(torch.all(torch.eq(a/b, torch.div(a,b))))

#矩阵相乘
#* element-wise 相同位置相乘
#. matrix mull 矩阵形式相乘
c = torch.rand(2,2)
d = torch.rand(2,2)
print(torch.mm(c,d))  #只能二维数据使用
print(torch.matmul(c,d))
print(c@d)
#矩阵降维
e = torch.rand(4,784)
f = torch.rand(4,784)
w = torch.rand(512,784)  #chanel-out chanel-in
print((f@w.t()).shape)  #torch.Size([4, 512])
#多维数据相乘
g = torch.rand(4,3,28,64)
h = torch.rand(4,3,64,32)
print(torch.matmul(g,h).shape)  #torch.Size([4, 3, 28, 32]) 只取最后两维矩阵相乘
i = torch.rand(4,1,64,32)
print(torch.matmul(g,i).shape)  #torch.Size([4, 3, 28, 32])

#pow
k = torch.full([2,2],3.)
print(k.pow(2))  #tensor([[9., 9.], [9., 9.]])
print(k**2)   #tensor([[9., 9.], [9., 9.]])
print(k.sqrt())  #平方根
print(k.rsqrt())  #平方根的倒数
print(k**(0.5))  #开方

#exp log
l = torch.exp(torch.ones(2,2))
print(torch.log(l))  #tensor([[1., 1.],[1., 1.]])

#approximation
m = torch.tensor(3.14)
print(m.floor(),m.ceil(),m.trunc(),m.frac(),m.round())  #tensor(3.) tensor(4.) tensor(3.) tensor(0.1400) tensor(3.) 向下取整，向上取整，裁剪整数部分，裁剪小数部分，四舍五入

#clamp
grad = torch.rand(2,3)*15
print(grad.max(),grad.median())  #tensor(13.5071) tensor(5.4121) 最大值，中间值
print(grad.clamp(10))  #tensor([[10.0000, 13.4692, 10.0000],[12.3968, 13.5071, 10.0000]])小于10变为10
print(grad.clamp(0,10))  #tensor([[ 0.0967, 10.0000,  5.4121],[10.0000, 10.0000,  1.6904]])大于10的变为10