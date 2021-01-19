import torch

#norm范数
a = torch.full([8], 1.)
b = a.view(2,4)
c = a.view(2,2,2)

print(a.norm(1),b.norm(1),c.norm(1))  #L1范数是指向量中各个元素绝对值之和。tensor(8.) tensor(8.) tensor(8.)
print(a.norm(2),b.norm(2),c.norm(2))  #范数是指向量各元素的平方和然后求平方根.tensor(2.8284) tensor(2.8284) tensor(2.8284)
print(b.norm(1, dim=1)),b.norm(2, dim=1)  #tensor([4., 4.])
print(c.norm(1, dim=0),c.norm(2, dim=0))  #tensor([[2., 2.], [2., 2.]]) tensor([[1.4142, 1.4142], [1.4142, 1.4142]])

#argmax...dim..
e = torch.arange(8).view(2,4).float()
print(e.min(),e.max(),e.mean(),e.prod(),e.sum())  #tensor(0.) tensor(7.) tensor(3.5000) tensor(0.) tensor(28.)
print(e.argmax(),e.argmin())  #返回索引，铺平后的索引 tensor(7) tensor(0)
f = torch.randn(4,10)
print(f.argmax(dim=1))  #指定维度返回索引 tensor([9, 9, 3, 2])
print(f.max(dim=1))  #指定维度返回值和索引 values=tensor([2.4001, 1.6390, 1.7758, 0.6443]),indices=tensor([4, 0, 2, 8]))
print(f.max(dim=1,keepdim=True))  #keepdim保留维度values=tensor([[1.1621],[2.0576],[1.8765], [1.5806]]),indices=tensor([[2],[1], [7], [7]]))

#top-k or k-th
print(f)
print(f.topk(3,dim=1))  #torch.return_types.topk(values=tensor([[0.9999, 0.9801, 0.9684],[1.9451, 1.0418, 1.0126], [1.2147, 0.7388, 0.5086], [1.8028, 1.4564, 0.4915]]),
                        #indices=tensor([[7, 9, 6],[5, 6, 1],[7, 2, 5], [4, 6, 1]]))
print(f.topk(3,dim=1,largest=False))  #largest=False最不可能的。torch.return_types.topk(values=tensor([[-1.2122, -0.9602, -0.4231], [-1.9420, -1.8693, -0.9472],[-1.0558, -1.0010, -0.7092],[-2.1212, -1.5748, -1.5341]]),
                                        #indices=tensor([[5, 0, 1], [8, 0, 4],[0, 8, 6], [5, 8, 3]]))
print(f.kthvalue(8,dim=1))  #概率第8小的值torch.return_types.kthvalue(values=tensor([0.9684, 1.0126, 0.5086, 0.4915]),indices=tensor([6, 1, 5, 1]))
print(f.kthvalue(3))  #torch.return_types.kthvalue(values=tensor([-0.4231, -0.9472, -0.7092, -1.5341]),indices=tensor([1, 4, 6, 3]))
print(f.kthvalue(3,dim=1))  #torch.return_types.kthvalue(values=tensor([-0.4231, -0.9472, -0.7092, -1.5341]),indices=tensor([1, 4, 6, 3]))

#compare
print(f>0.5)  #大于为True
print(torch.gt(f,0))  #torch.gt(a,b)函数比较a中元素大于（这里是严格大于）b中对应元素，大于则为1，不大于则为0
g = torch.ones(2,3)
h = torch.randn(2,3)
print(torch.eq(g,h))  #2*3False
print(torch.eq(g,g)) #2*3True
print(torch.equal(g,g)) #True