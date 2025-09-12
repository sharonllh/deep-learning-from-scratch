import numpy as np
from loss_function import cross_entropy_error
from activation_function import softmax
from numerical_gradient import numerical_gradient

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # 随机生成权重W，形状为2*3，每个元素的值服从标准正态分布N(0,1)
    
    def predict(self, x):
        z = np.dot(x, self.W)
        y = softmax(z)
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        loss = cross_entropy_error(y, t)
        return loss

def test():
    net = SimpleNet()
    print("权重:")
    print(net.W)

    x = np.array([0.6, 0.9])
    y = net.predict(x)
    print("预测值:")
    print(y)

    t = np.array([0, 0, 1])
    loss = net.loss(x, t)
    print("损失函数:")
    print(loss)

    f = lambda w: net.loss(x, t)
    dW = numerical_gradient(f, net.W)
    print("梯度:")
    print(dW)

if __name__ == '__main__':
    test()