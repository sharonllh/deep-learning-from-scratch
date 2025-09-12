import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from dataset.mnist import load_mnist
from multi_layer_net import MultiLayerNet

# 随机梯度下降法
class SGD:
    # lr: learning rate，学习率
    def __init__(self, lr=0.01):
        self.lr = lr

    # params: 权重
    # grads: 梯度
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

# 动量法
class Momentum:
    # lr: 学习率
    # momentum: 摩擦力
    # v: 速度
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


# AdaGrad法
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key] # x * x: 把数组的逐个元素相乘
            params[key] -= self.lr * grads[key] / np.sqrt(self.h[key] + 1e-7) # 加1e-7是为了防止除以0的情况


# Adam法，结合了Momentum和AdaGrad的方法
class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)


def f(x, y):
    return x**2 / 20.0 + y**2

def df(x, y):
    return x / 10.0, 2.0 * y

# 最优化方法比较: 基于f(x, y)函数
def compare_optimizer_naive():
    init_pos = (-7.0, 2.0)
    params = {}
    params['x'], params['y'] = init_pos[0], init_pos[1]
    grads = {}
    grads['x'], grads['y'] = 0, 0

    optimizers = OrderedDict()
    optimizers['SGD'] = SGD(lr=0.95)
    optimizers['Momentum'] = Momentum(lr=0.1)
    optimizers['AdaGrad'] = AdaGrad(lr=1.5)
    optimizers['Adam'] = Adam(lr=0.3)

    idx = 1
    for key in optimizers:
        optimizer = optimizers[key]
        x_history = []
        y_history = []
        params['x'], params['y'] = init_pos[0], init_pos[1]

        for i in range(30):
            x_history.append(params['x'])
            y_history.append(params['y'])
            grads['x'], grads['y'] = df(params['x'], params['y'])
            optimizer.update(params, grads)

        x = np.arange(-10, 10, 0.01)
        y = np.arange(-5, 5, 0.01)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)

        plt.subplot(2, 2, idx)
        idx += 1
        plt.plot(x_history, y_history, 'o-', color='red')
        plt.contour(X, Y, Z) # 画Z = f(X, Y)的等高线
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.plot(0, 0, '+')
        plt.title(key)
        plt.xlabel('x')
        plt.ylabel('y')

    plt.show()

# 最优化方法比较：基于mnist数据集
def compare_optimizer_mnist():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    train_size = x_train.shape[0]
    batch_size = 128
    max_iterations = 3000

    optimizers = OrderedDict()
    optimizers['SGD'] = SGD()
    optimizers['Momentum'] = Momentum()
    optimizers['AdaGrad'] = AdaGrad()
    optimizers['Adam'] = Adam()

    networks = {}
    train_loss = {}
    for key in optimizers.keys():
        networks[key] = MultiLayerNet(
            input_size=784,
            hidden_size_list=[100, 100, 100, 100], 
            output_size=10)
        train_loss[key] = []
    
    for i in range(max_iterations):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for key in optimizers.keys():
            network = networks[key]
            grads = network.gradient(x_batch, t_batch)
            params = network.params
            optimizers[key].update(params, grads)

            loss = network.loss(x_batch, t_batch)
            train_loss[key].append(loss)
        
        if i % 100 == 0:
            print("-----" + "iteraction: " + str(i) + "-----")
            for key in optimizers.keys():
                loss = networks[key].loss(x_batch, t_batch)
                print(key + ": " + str(loss))
    
    plt.figure()
    markers = {
        "SGD": "o",
        "Momentum": "x",
        "AdaGrad": "s",
        "Adam": "D"
    }
    x = np.arange(max_iterations)
    for key in optimizers.keys():
        plt.plot(x, train_loss[key], marker=markers[key], markevery=10, label=key) # markevery用于稀疏地显示marker，这里设置的是每10个点显示一个marker
    plt.xlabel('iterations')
    plt.ylabel('loss')
    #plt.ylim(0, 1)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    compare_optimizer_naive() # AdaGrad > Adam > Momentum > SGD
    compare_optimizer_mnist() # AdaGrad > Adam > Momentum > SGD