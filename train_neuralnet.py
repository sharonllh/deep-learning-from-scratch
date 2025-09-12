import numpy as np
import matplotlib.pyplot as plt
from two_layer_net_with_backprop import TwoLayerNet
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 超参数
iters_num = 2000
train_size = x_train.shape[0]
batch_size = 128
learning_rate = 0.01

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 平均每个epoch的重复次数
# 一个epoch表示所有训练数据都被使用过一次的更新次数，例如：对于5000个训练数据，每个batch为100，那么一个epoch就是50次
#iter_per_epoch = max(train_size / batch_size, 1)
iter_per_epoch = 20

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    # grad = network.numerical_gradient(x_batch, t_batch) # 运行超慢！
    grad = network.gradient(x_train, t_train)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 记录损失
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(str(i) + " | train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 绘制学习过程
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(np.arange(iters_num), train_loss_list)
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(1, 2, 2)
epoch = np.arange(len(train_acc_list))
plt.plot(epoch, train_acc_list, label='train acc')
plt.plot(epoch, test_acc_list, label='test acc', linestyle='--')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()

plt.show()
