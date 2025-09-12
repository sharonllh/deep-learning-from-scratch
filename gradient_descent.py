import numpy as np
import matplotlib.pylab as plt
from numerical_gradient import numerical_gradient, function_2

# 梯度下降法：求解函数的极小值
# 参数：
#    f：待优化的函数
#    init_x：初始值
#    lr: 学习率learning rate
#    step_num：梯度法的重复次数
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = [] # 用于绘图

    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x, np.array(x_history)

def test():
    print("------用梯度法求function_2的最小值-----")
    init_x = np.array([-3.0, 4.0])
    min, x_history = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
    print("初始值为(-3,4), lr=0.1:")
    print(min) # [-6.11110793e-10  8.14814391e-10]

    print("-----绘制梯度法求解的过程-----")
    plt.plot([-5,5], [0,0], '--b') # 绘制一条从(-5,0)搭配(5,0)的蓝色虚线
    plt.plot([0,0], [-5,5], '--b')
    plt.plot(x_history[:,0], x_history[:,1], 'o') # 以x_history的第0列为横坐标、第1列为纵坐标，绘制散点图
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.show()

    print("-----学习率过大或过小时-----")
    print("lr=10.0:")
    init_x = np.array([-3.0, 4.0]) # 必须重新赋值，之前的值会被更改
    min, _ = gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100)
    print(min) # [-2.58983747e+13 -1.29524862e+12]

    print("lr=1e-10:")
    init_x = np.array([-3.0, 4.0])
    min, _ = gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100)
    print(min) # [-2.99999994  3.99999992]

if __name__ == '__main__':
    test()