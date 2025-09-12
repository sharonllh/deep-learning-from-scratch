#import sys, os
#sys.path.append(os.pardir) # 导入模块时，会去搜索父目录
import numpy as np
from dataset.mnist import load_mnist #从dataset目录的mnist.py中导入load_mnist函数
from PIL import Image
import pickle
from activation_function import sigmoid, softmax

### 了解mnist数据的结构 ###

# 显示numpy数组表示的图像
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 了解mnist数据的结构
def learn_mnist_data():
    # 读取MNIST数据集，第一次运行时会下载数据集到本地
    # 返回：(训练图像, 训练标签), (测试图像, 测试标签)
    # 参数：
    #     flatten: 图像原本是1*28*28的三维数组，是否将其变为784个元素构成一维数组
    #     normalize: 像素值原本为0~255，是否将其归一化到0.0~1.0
    #     one_hot_label: 标签原来是0~9的数字，是否将其变成one-hot表示，即2变成[0,0,1,0,0,0,0,0,0,0]
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

    # 输出数据的形状
    # 6w个训练数据，1w个测试数据
    print("-----shape of mnist data-----")
    print(x_train.shape) # (60000, 784)
    print(t_train.shape) # (60000,)
    print(x_test.shape) # (10000, 784)
    print(t_test.shape) # (10000,)

    # 显示第一个训练数据
    img = x_train[0]
    label = t_train[0]
    print("-----first train label-----")
    print(label) # 5

    print("-----first train image-----")
    print(img.shape) # (784,)
    img = img.reshape(28, 28) # 把图像变成原来的尺寸
    print(img.shape) # (28, 28)

    img_show(img)

print("------learn mnist data------")
learn_mnist_data()

### 用训练好的神经网络来分类 ###

# 获取测试数据集
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

# 获取训练好的神经网络
# 输入层x: 784个神经元
# 输出层y: 10个神经元
# 隐藏层z1: 50个神经元
# 隐藏层z2: 100个神经元
def init_network():
    with open(r"dataset\sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    
    return network

# 对输入进行分类
# 输出：数字0-9的概率
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

# 每次对一张图片进行分类
def handle(network, x, t):
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y) # 获取概率最高的元素的索引
        if p == t[i]:
            accuracy_cnt += 1

    accuracy = float(accuracy_cnt) / len(x)
    print("------handle------")
    print("Accuracy: " + str(accuracy))

# 每次对多张图片进行批量处理
def batch_handle(network, x, t):
    batch_size = 100
    accuracy_cnt = 0
    print("------batch handle------")
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        #print(x_batch.shape) # (100,784)
        y_batch = predict(network, x_batch)
        #print(y_batch.shape) # (100, 10)
        p = np.argmax(y_batch, axis=1) # 按行找最大元素的索引
        #print(p.shape) # (100,)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])
    
    accuracy = float(accuracy_cnt) / len(x)
    print("Accuracy: "+ str(accuracy))

# 测试
print("------nerual network predict------")
x, t = get_data()
print("------shape of input------")
print(x.shape) # (10000, 784)
print(x[0].shape) # (784,)

network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']
print("------shape of weight------")
print(W1.shape) # (784, 50)
print(W2.shape) # (50, 100)
print(W3.shape) # (100, 10)

handle(network, x, t)
batch_handle(network, x, t)