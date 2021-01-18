import torch


a = torch.rand(4,1,28,28)
a.view(4,28*28).shape   #合并通道，要有物理意义的合并，不然会污染数据
a.view(4*28,28).shape
a.view(4*1,28,28).shape
b = a.view(4,784)  #b不能知道a真实维度

#增加维度
print(a.shape)  #torch.Size([4, 1, 28, 28])
print(a.unsqueeze(0).shape)  #torch.Size([1, 4, 1, 28, 28])  在某个位置之后插入新维度，不会改变数据本身
print(a.unsqueeze(-1).shape)  #torch.Size([4, 1, 28, 28, 1])
x = torch.tensor([1.2,2.3])
print(x.unsqueeze(-1))  #tensor([[1.2000],[2.3000]])增加了维度
print(x)  #tensor([1.2000, 2.3000])本身未变
#将z变化与f相同
z = torch.rand(32)
f = torch.rand(4,32,14,14)
z = z.unsqueeze(1).unsqueeze(2).unsqueeze(0)
print(z.shape)  #torch.Size([1, 32, 1, 1])

#挤压维度
print(z.squeeze().shape)    #torch.Size([32])挤压维度,无参数：能挤压的都挤压
print(z.squeeze(1).shape)  #torch.Size([1, 32, 1, 1]) 不是1不能挤压
print(z.squeeze(-1).shape)  #torch.Size([1, 32, 1])

print(z.shape)  #torch.Size([1, 32, 1, 1])
print(f.shape)  #torch.Size([4, 32, 14, 14])
print(z.expand(4,32,14,14).shape)  #torch.Size([4, 32, 14, 14]) 只有1维才能复制扩充,非1维要一致
print(z.expand(-1,32,14,-1).shape)  #torch.Size([1, 32, 14, 1]) -1：保留维度
print(z.repeat(4,32,14,14).shape)  #torch.Size([4, 1024, 14, 14])  参数为每个维度重复的次数

#转置
w = torch.randn(3,4)
print(w)  #tensor([[ 0.7477, -0.1726,  0.7004,  1.6182], [ 0.3078, -0.8410,  0.5541,  0.1974],[ 1.3985,  0.8547,  0.4088, -1.6937]])
print(w.t())  #转置，只能用于二维矩阵 tensor([[ 0.7477,  0.3078,  1.3985], [-0.1726, -0.8410,  0.8547], [ 0.7004,  0.5541,  0.4088], [ 1.6182,  0.1974, -1.6937]])

#交换维度
m = torch.rand(4,3,32,32)
mm = m.transpose(1,3).contiguous().view(4,3*32*32).view(4,32,32,3).transpose(1,3)  #跟踪住先后顺序  contiguous()保证连续
print(mm.shape)  #torch.Size([4, 3, 32, 32])
print(m.permute(0,2,3,1).shape)  #torch.Size([4, 32, 32, 3])  交换为0，2，3，1通道