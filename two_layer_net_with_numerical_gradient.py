import numpy as np
from activation_function import sigmoid, softmax
from loss_function import cross_entropy_error
from numerical_gradient import numerical_gradient

# 用数值微分法来计算梯度的两层神经网络
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) # 用符合高斯分布的随机数进行初始化
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1) # 对每一行求最大值的索引
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
    
def test():
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print("-----shape of params-----")
    print("W1:")
    print(net.params['W1'].shape) # (784, 100)
    print("b1:")
    print(net.params['b1'].shape) # (100,)
    print("W2:")
    print(net.params['W2'].shape) # (100, 10)
    print("b2:")
    print(net.params['b2'].shape) # (10,)

    x = np.random.rand(5, 784) # 生成5*784的二维数组，每个元素的值服从[0,1)均匀分布
    t = np.random.rand(5, 10)

    print("-----predict-----")
    y = net.predict(x)
    print(y)

    print("-----loss-----")
    loss = net.loss(x, t)
    print(loss)

    print("-----numerical gradient-----")
    grads = net.numerical_gradient(x, t)
    print(grads)


if __name__ == '__main__':
    test()
        
