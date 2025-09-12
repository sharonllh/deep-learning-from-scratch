import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from multi_layer_net import MultiLayerNet
from trainer import Trainer

# 打乱数据集
def shuffle_dataset(x, t):
    permutation = np.random.permutation(x.shape[0]) # 生成一个打乱的索引序列，如np.random.permutation(5)返回[0, 3, 4, 1, 2]
    x = x[permutation, :]
    t = t[permutation]
    return x, t


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

x_train, t_train = x_train[:500], t_train[:500]

validation_rate = 0.2
validation_num = int(x_train.shape[0] * validation_rate)

x_train, t_train = shuffle_dataset(x_train, t_train)
x_val, t_val = x_train[:validation_num], t_train[:validation_num] # 验证数据，用来验证超参数
x_train, t_train = x_train[validation_num:], t_train[validation_num:] # 训练数据，用来学习


def train(lr, weight_decay, epochs=50):
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                            use_weight_decay=True, weight_decay_lambda=weight_decay)
    
    trainer = Trainer(network, x_train, t_train, x_val, t_val, 
                      epochs=epochs, batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': lr},
                      verbose=False)
    
    trainer.train()

    return trainer.train_acc_list, trainer.test_acc_list


optimization_trial = 100
results_train = {}
results_val = {}

for i in range(optimization_trial):
    weight_decay = 10 ** np.random.uniform(-8, -4) # 随机生成一个[10^-8, 10^-4)之间的随机数
    lr = 10 ** np.random.uniform(-6, -2)

    train_acc_list, val_acc_list = train(lr, weight_decay)
    key = "lr:" + str(lr) + " | weight_decay:" + str(weight_decay)
    results_train[key] = train_acc_list
    results_val[key] = val_acc_list

    print(key + " | val_acc:" + str(val_acc_list[-1]))

# 画图
graph_num = 20
col_num = 5
row_num = int(np.ceil(graph_num / col_num))
i = 1

for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True): # 按照results_val的值的最后一个元素降序排序
    print("Best-" + str(i) + " | " + key + " | val_acc:" + str(val_acc_list[-1]))

    plt.subplot(row_num, col_num, i)
    plt.title("Best-" + str(i))
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    plt.xticks([]) # x轴不显示刻度

    if i >= graph_num:
        break
    i += 1

plt.show()