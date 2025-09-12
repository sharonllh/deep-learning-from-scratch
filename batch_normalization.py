import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist
from multi_layer_net import MultiLayerNet
from optimizer import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01
iter_per_epoch = max(train_size / batch_size, 1)

def train(weight_init_std):
    bn_network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100], output_size=10, weight_init_std=weight_init_std, use_batch_norm=True)
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100], output_size=10, weight_init_std=weight_init_std, use_batch_norm=False)
    optimizer = SGD(lr=learning_rate)

    bn_train_acc_list = []
    train_acc_list = []
    epoch_cnt = 0

    for i in range(100000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 更新权重
        for net in (bn_network, network):
            grads = net.gradient(x_batch, t_batch)
            optimizer.update(net.params, grads)
        
        # 计算准确度
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print("epoch: " + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))

            epoch_cnt += 1
            if (epoch_cnt >= max_epochs):
                break

    return train_acc_list, bn_train_acc_list

# 画图
weight_init_styles = {
    'std=0.01': 0.01,
    'std=0.2': 0.2,
    'Xavier': 'Xavier',
    'He': 'He'
}

x = np.arange(max_epochs)
plt.figure()

len = len(weight_init_styles)
idx = 1
for key, weight_init_std in weight_init_styles.items():
    train_acc_list, bn_train_acc_list = train(weight_init_std=weight_init_std)

    plt.subplot(1, len, idx)
    plt.plot(x, train_acc_list, linestyle='--', label='Normal (without BatchNorm)')
    plt.plot(x, bn_train_acc_list, label='Batch Normalization')
    plt.legend()
    plt.title(key)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")

    idx += 1

plt.show() # 对于绝大部分的权重初始值，使用Batch Normalization都能使学习进行得更快