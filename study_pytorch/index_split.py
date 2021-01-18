import torch


a = torch.rand(4,3,28,28)  #batch,chanel,height,weigh
print(a[0].shape)  #torch.Size([3, 28, 28]) 第一张图片
print(a[0,0].shape)  #torch.Size([28, 28]) 第0图片第0通道
print(a[0,0,2,4].shape)  #torch.Size([]) 第0图片第0通道第2行第4列
print(a[:2].shape)  #0-2(不包含2)
print(a[:2,:1,:,:].shape)  #:,0-最后全取
print(a[:2,1:,:,:].shape)  #1:,第1到最末尾
print(a[:2,-1:,:,:].shape)  #-1：，最末尾到全部
print(a[:,:,0:28:2,0:28:2].shape)  #0：28：2隔行采样，0-28（不包含28），2步长
print("---------------------------------------------")
print(a.index_select(3,torch.arange(8)).shape)
print(a.index_select(2,torch.arange(28)).shape)  #dim,index

print(a[...].shape)  #a，。。。推测取多少
print(a[0,...].shape)  #。。。每一个维度都取
print(a[0,...,::2].shape)  #确定了第一个最后一个，所以中间表示两个得所有

print("---------------------------------------------")
x = torch.randn(3, 4)
mask = x.get(0.5)  #铺平，大于等于0.5的位置标为1（int）
torch.masked_select(x, mask)  #取出大于0.5的值取出来

src = torch.tensor([[4,3,5],[6,7,8]])
torch.take(src, torch.tensor([0,2,5]))  #铺平，用索引取