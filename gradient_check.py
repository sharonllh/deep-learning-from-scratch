import numpy as np
from two_layer_net_with_backprop import TwoLayerNet
from dataset.mnist import load_mnist

# 梯度确认：比较数值微分得到的梯度和误差反向传播法得到的梯度是否一致，以确认误差反向传播法的实现是否正确
def gradient_check(batch_size):
    print("------batch size=" + str(batch_size) + "-----")
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    x_batch = x_train[:batch_size]
    t_batch = t_train[:batch_size]

    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)

    # 求每个权重的梯度误差
    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(key + ": " + str(diff))

if __name__ == '__main__':
    gradient_check(batch_size=10) # 误差很小，如W1: 2.6510968151369885e-10
    #gradient_check(batch_size=100)