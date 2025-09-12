import numpy as np
import matplotlib.pylab as plt

# 阶跃函数 - 输入实数
# def step_function(x):
#     if x > 0:
#         return 1
#     else:
#         return 0

# 阶跃函数 - 输入numpy数组    
# def step_function(x):
#     y = x > 0 # y是bool数组
#     return y.astype(int) # 转成int数组

# 阶跃函数
def step_function(x):
    return np.array(x > 0, dtype=int)

# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU函数
def relu(x):
    return np.maximum(x, 0)

# 恒等函数
def identify_function(x):
    return x

# softmax函数 - 有溢出问题
# def softmax(x):
#     exp_x = np.exp(x)
#     sum_exp_x = np.sum(exp_x)
#     y = exp_x / sum_exp_x
#     return y

# softmax函数
def softmax(x):
    if x.ndim == 1:
        x = x.reshape(1, -1) # 把一维数组变成二维数组

    c = np.max(x, axis=1, keepdims=True) # 对每一行找最大值
    exp_x = np.exp(x - c) # 解决溢出问题
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    y = exp_x / sum_exp_x
    return y

def test():
    print("---测试阶跃函数---")
    x = np.array([-1.0, 1.0, 2.0])
    y = step_function(x)
    print(y)

    print("---测试sigmoid函数---")
    x = np.array([-1.0, 1.0, 2.0])
    y = sigmoid(x)
    print(y)

    print("---测试ReLU函数---")
    x = np.array([-1.0, 1.0, 2.0])
    y = relu(x)
    print(y)

    print("---测试softmax函数---")
    x = np.array([0.3, 2.9, 4.0])
    y = softmax(x)
    print(y)
    print(np.sum(y))

    # 绘制激活函数
    x = np.arange(-10.0, 10.0, 0.1)
    y1 = step_function(x)
    y2 = sigmoid(x)
    y3 = relu(x)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8)) # 2行2列

    axs[0, 0].plot(x, y1, label="step", linestyle="--")
    axs[0, 0].plot(x, y2, label="sigmoid")
    #plt.ylim(-0.1, 1.1) # y轴的范围
    axs[0, 0].set_title("step & sigmoid")
    axs[0, 0].legend()

    axs[0, 1].plot(x, y3)
    axs[0, 1].set_title("LeLU")

    plt.show()

if __name__ == '__main__':
    test()
