import numpy as np
import matplotlib.pyplot as plt
from multi_layer_net import MultiLayerNet
from trainer import Trainer
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

x_train = x_train[:300]
t_train = t_train[:300]

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10, use_dropout=False)
network_dropout = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10, use_dropout=True, dropout_ratio=0.2)

train_acc_list=[]
test_acc_list=[]

for idx, net in enumerate([network, network_dropout]):
    trainer = Trainer(net, x_train, t_train, x_test, t_test, 
                    epochs=300, batch_size=100, 
                    optimizer='sgd', optimizer_param={'lr':0.01},
                    verbose=True)

    trainer.train()
    train_acc_list.append(trainer.train_acc_list)
    test_acc_list.append(trainer.test_acc_list)

# 画图
for idx in range(2):
    plt.subplot(1, 2, idx + 1)
    x = np.arange(len(train_acc_list[idx]))
    plt.plot(x, train_acc_list[idx], marker='o', label='train acc', markevery=10)
    plt.plot(x, test_acc_list[idx], marker='s', label='test acc', markevery=10)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    
    title = "No dropout" if idx == 0 else "Dropout"
    plt.title(title)

plt.show() # 使用dropout后，训练数据和测试数据的识别精度相差更小，并且训练数据的识别精度达不到100%