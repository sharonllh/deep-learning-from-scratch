import numpy as np
import matplotlib.pyplot as plt
from activation_function import sigmoid, relu

input_data = np.random.randn(1000, 100) # 1000个输入数据
node_num = 100 # 每个隐藏层有100个节点
hidden_layer_size = 5 # 5个隐藏层

# 绘制激活值的直方图
# f: 激活函数
# std: 高斯分布的标准差
def draw_activation_histogram(f, std):
    x = input_data
    activations = {} # 每个隐藏层的激活值

    for i in range(hidden_layer_size):
        if i != 0:
            x = activations[i - 1]
        
        w = np.random.randn(node_num, node_num) * std

        a = np.dot(x, w)
        z = f(a)
        activations[i] = z

    # 绘制直方图
    plt.figure()
    for i, z in activations.items():
        plt.subplot(1, len(activations), i+1)
        plt.title(str(i+1) + "-layer")
        plt.hist(z.flatten(), 30, range=(0,1)) # 统计z在区间[0,1]内的值的分布情况，把数据分成30个区间

    plt.suptitle(f.__name__ + ", std=" + str(std))
    plt.show()

if __name__ == '__main__':
    draw_activation_histogram(f=sigmoid, std=1)
    draw_activation_histogram(f=sigmoid, std=0.01)
    draw_activation_histogram(f=sigmoid, std=1.0/np.sqrt(node_num)) # Xavier初始值，推荐！

    draw_activation_histogram(f=relu, std=0.01)
    draw_activation_histogram(f=relu, std=1.0/np.sqrt(node_num)) # Xavier初始值
    draw_activation_histogram(f=relu, std=np.sqrt(2.0/node_num)) # He初始值，推荐！
