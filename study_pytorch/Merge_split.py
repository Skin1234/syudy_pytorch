import torch

#concat
a = torch.rand(4,32,8)
b = torch.rand(5,32,8)
print(torch.cat([a,b],dim=0).shape)  #torch.Size([9, 32, 8])  [a,b]:合并哪些数据 dim=0：在第几维度合并  .只有在dim维度数字不同，其他必须相同

#stack
c = torch.rand(32,8)
d = torch.rand(32,8)
print(torch.stack([c,d],dim=0).shape)  #torch.Size([2, 32, 8])  创建新维度，合并。shape完全一致

#split
aa,bb = a.split(3,dim=0)  #torch.Size([3, 32, 8]) torch.Size([1, 32, 8]) 固定一块的个数
print(aa.shape,bb.shape)
cc= b.chunk(3,dim=0)  #torch.Size([2, 32, 8]) torch.Size([2, 32, 8]) torch.Size([1, 32, 8]) 将tensor按dim方向分割成chunks个tensor块，返回的是一个元组,或者三个张量。
print(cc)
dd,ee,ff= b.chunk(3,dim=0)
print(dd.shape,ee.shape,ff.shape)