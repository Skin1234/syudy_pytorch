import torch
from torch import autograd
import numpy as np

torch.FloatTensor([2.,3.2])  #现有数据
torch.FloatTensor(2,3)  #维度
torch.tensor([2.,3.2])  #现有数据
torch.tensor([[2.,4.],[5.2,6.3]])  #生成得数据跨度大，要初始化覆盖掉，不然会有很多问题

#未初始化
'''
torch.empty([2,3])
torch.FloatTensor(d1,d2,d3)
torch.IntTensor(d1,d2,d3)
'''

#随机初始化
a = torch.rand(3,3)  #在0~1之间随机生成均匀数
torch.rand_like(a) #随机sample,参数是list
torch.randint(1,10,[3,3])  #最小值，最大值，shape  [min.max)
torch.randn(3,3)   #j均值0，方差1，随机取值
torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1)) #full生成均值`长度为10都为0的list,方差是1-0慢慢减少
torch.normal(mean=0.5, std=torch.arange(1., 6.))

torch.full([2,3],7)  #2*3tensor,值全部为7,默认类型
torch.full([],7)  #[]=shape
torch.full([1],7)  #[1] = 1*1 [2] = [7.,7.]
torch.arange(0,10)  #默认1，等差数列
torch.arange(0,10,2)  #等差为2
#torch.range(0,10)  #不建议pytorch使用
print(torch.linspace(0,10,steps=4))  #steps切割数量 tensor([ 0.0000,  3.3333,  6.6667, 10.0000])
print(torch.linspace(0,10,steps=10))  #tensor([ 0.0000,  1.1111,  2.2222,  3.3333,  4.4444,  5.5556,  6.6667,  7.7778, 8.8889, 10.0000])
print(torch.linspace(0,10,steps=11))  #tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
print(torch.linspace(0,-1,steps=10))  #tensor([ 0.0000, -0.1111, -0.2222, -0.3333, -0.4444, -0.5556, -0.6667, -0.7778, -0.8889, -1.0000])

torch.ones(3,3)  #3*3全1
torch.zero_(3,3)  #3*3全0
torch.eye(3,4)  #3*4对角为1，不是正方形的多余列为0.
torch.eye(3)  #只能接受1/2参数，只能用来创建二维矩阵
z = torch.zeros(3,3)
torch.ones_like(z)  #和z相同形状全1

torch.randperm(10)  #0-9随机生成10个索引