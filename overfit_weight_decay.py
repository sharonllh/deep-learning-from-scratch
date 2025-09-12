import numpy as np
import matplotlib.pyplot as plt
from multi_layer_net import MultiLayerNet
from dataset.mnist import load_mnist
from optimizer import SGD

# 为了制造过拟合：
# 1. 增加神经网络的复杂度
# 2. 减少训练数据

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

x_train = x_train[:300]
t_train= t_train[:300]

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10, use_weight_decay=False)
network_wd = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10, use_weight_decay=True, weight_decay_lambda=0.1)
optimizer = SGD(lr=0.01)

max_epochs = 200
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = max(train_size / batch_size, 1)

epoch_cnt = 0
train_acc_list = []
test_acc_list = []
train_acc_list_wd = []
test_acc_list_wd = []

for i in range(100000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    grads_wd = network_wd.gradient(x_batch, t_batch)
    optimizer.update(network_wd.params, grads_wd)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        train_acc_wd = network_wd.accuracy(x_train, t_train)
        test_acc_wd = network_wd.accuracy(x_test, t_test)
        train_acc_list_wd.append(train_acc_wd)
        test_acc_list_wd.append(test_acc_wd)

        print(str(i) + ": " + str(train_acc) + " - " + str(test_acc) + " | " + str(train_acc_wd) + " - " + str(test_acc_wd))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

# 画图
x = np.arange(max_epochs)
plt.subplot(1, 2, 1)
plt.plot(x, train_acc_list, label="train acc", marker="o", markevery=20)
plt.plot(x, test_acc_list, label="test acc", marker="s", markevery=20)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.title("No Weight Decay") # 当epochs超过100后，训练数据的识别精度已经接近甚至达到100%了，但是测试数据的识别精度还是比较低

plt.subplot(1, 2, 2)
plt.plot(x, train_acc_list_wd, label="train acc", marker="o", markevery=20)
plt.plot(x, test_acc_list_wd, label="test acc", marker="s", markevery=20)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.title("Weight Decay") # 训练数据和测试数据的识别精度差距更小，过拟合得到了抑制，另外训练数据的识别精度没有达到100%

plt.show() 
