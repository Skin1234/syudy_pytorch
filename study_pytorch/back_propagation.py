import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

#找全局最小值
def himmelblau(x):  #一个神经元
    return (x[0]  ** 2 + x[1] - 11)  ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

#可视化
x = np.arange(-6, 6 , 0.1)  #构建等差数列做为坐标轴
y = np.arange(-6, 6, 0.1)
print('x, y range:', x.shape, y.shape)
X, Y = np.meshgrid(x, y)
print('X, Y maps:', X.shape, Y.shape)
Z = himmelblau([X, Y])

fig = plt.figure('himmelblau')  #获取到当前figure对象。
ax = fig.gca(projection='3d')  #fig.gca是获取图中的当前极轴。如果不存在，或者不是极轴，则将创建相应的轴，然后返回。
ax.plot_surface(X,Y,Z)  #绘制三维曲面
ax.view_init(60, -30)  #设置视角
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

#x = torch.tensor([0., 0.], requires_grad=True)  #x = [3.0, 2.0]
x = torch.tensor([-4., 0.], requires_grad=True)  #设置不同的初始化，得到不同的最小值 x = [-3.7793102264404297, -3.2831859588623047]
optimizer = torch.optim.Adam([x], lr=1e-3)  #设置迭代优化算法，参数：迭代数列，学习速率
for step in range(20000):

    pred = himmelblau(x)  #优化预测值而不是loss,优化x

    optimizer.zero_grad()  #优化器清零
    pred.backward()  #预测值反向求偏导,得到x,y的梯度信息
    optimizer.step()  #调用优化算法

    if step % 2000 == 0:  #每2000次迭代打印一次信息
        print('step {}: x = {}, f(x) = {}'.format(step, x.tolist(), pred.item()))