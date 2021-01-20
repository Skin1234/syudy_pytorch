import torch
import torch.nn.functional as F
#loss for classification:MSE,Cross Entropy Loss,Hinge Loss

#Entropy
a = torch.full([4], 1/4.)

print(a*torch.log2(a))  #tensor([-0.5000, -0.5000, -0.5000, -0.5000])
print(-(a*torch.log2(a)).sum()) #Entropy:-p*log2(1/p)    tensor(2.)

a = torch.tensor([0.1,0.1,0.1,0.7])  #中奖率
print(-(a*torch.log2(a)).sum())  #不稳定 tensor(1.3568)

a = torch.tensor([0.001,0.001,0.001,0.999])
print(-(a*torch.log2(a)).sum())  #越小越不稳定 tensor(0.0313)

#Cross entroy = softmax+log+null_loss
x = torch.randn(1, 784)
w = torch.randn(10, 784)

logits = x@w.t()
pred = F.softmax(logits, dim=1)
pred_log = torch.log(pred)
F.cross_entropy(logits, torch.tensor([3]))
F.nll_loss(pred_log, torch.tensor([3]))