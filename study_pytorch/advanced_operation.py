import torch

#where
cond = torch.tensor([[0.6769, 0.7271],
                      [0.8884, 0.4163]])
a = torch.ones(2,2)
b = torch.zeros(2,2)
print(torch.where(cond>0.5, a, b))  #cond 大于0.5取a的值，小于取b的值tensor([[1., 1.],
                                                                        #[1., 0.]])

#gather 类似查表操作
prob = torch.randn(4,10)
idx = prob.topk(dim=1, k=3)  #最有可能的前三4*3值+4*3索引
idx= idx[1] #取出索引4*3
table = torch.arange(10)+100 #1*10的表
result = torch.gather(table.expand(4,10), dim=1, index=idx.long())  #1*10扩展成4*10， idx4*3的每行在table4*10每行查询
print(result)  #idx索引对应在table的值
