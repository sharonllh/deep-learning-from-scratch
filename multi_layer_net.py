import numpy as np
from collections import OrderedDict
from layers import *
from numerical_gradient import numerical_gradient
from dataset.mnist import load_mnist

class MultiLayerNet:
    def __init__(self, input_size, hidden_size_list, output_size, 
                 activation='relu', weight_init_std='relu',
                 use_batch_norm=False,
                 use_weight_decay=False, weight_decay_lambda=0,
                 use_dropout=False, dropout_ratio=0.5):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)

        # 是否使用batch normalization
        self.use_batch_norm = use_batch_norm

        # 是否使用权值衰减
        self.use_weight_decay = use_weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        # 是否使用dropout
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio

        self.params = {}

        # 初始化权重
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

        # 创建layers
        activation_layer = {
            'sigmoid': Sigmoid,
            'relu': Relu
        }
        self.layers = OrderedDict()

        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

            # 在每个Affine层后面添加BatchNorm层
            if self.use_batch_norm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])

            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

            # 在每一层后面添加Dropout层
            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(self.dropout_ratio)
        
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        loss = self.last_layer.forward(y, t)

        # 使用权值衰减：为损失函数加上权重的L2范数，即 1/2 * lambda * W^2
        if self.use_weight_decay:
            weight_decay = 0
            for idx in range(1, self.hidden_layer_num + 2):
                W = self.params['W' + str(idx)]
                weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
            loss += weight_decay

        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

            # 关于BatchNorm层权重的梯度
            if self.use_batch_norm and idx != self.hidden_layer_num + 1:
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])

        return grads
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # 梯度
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            # 使用权值衰减：为权重梯度加上lambda * W
            if self.use_weight_decay:
                grads['W' + str(idx)] += self.weight_decay_lambda * self.layers['Affine' + str(idx)].W

            # 关于BatchNorm层权重的梯度
            if self.use_batch_norm and idx != self.hidden_layer_num + 1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta
        
        return grads
    

def gradient_check(batch_size):
    print("------batch size=" + str(batch_size) + "-----")
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    batch_mask = np.random.choice(x_train.shape[0], batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    net = MultiLayerNet(input_size=784, hidden_size_list=[100, 100], output_size=10)
    grad_numerical = net.numerical_gradient(x_batch, t_batch)
    grad_backprop = net.gradient(x_batch, t_batch)

    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(key + ": " + str(diff))

if __name__ == '__main__':
    gradient_check(10) # 误差很小，如W1: 1.5213900212971665e-08
    #gradient_check(100)