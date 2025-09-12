import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from convolutional_neural_network import SimpleConvNet
from trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True, flatten=False)

max_epochs = 20
batch_size = 100

network = SimpleConvNet(input_dim=(1,28,28),
                        conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, batch_size=batch_size,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)

trainer.train()

# 保存权重
network.save_params("params.pkl")
print("Network parameters saved!")

# 画图
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', markevery=2, label='train acc')
plt.plot(x, trainer.test_acc_list, marker='s', markevery=2, label='test acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show() # test acc: 98.61%