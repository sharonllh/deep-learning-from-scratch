import numpy as np
import matplotlib.pylab as plt

# 数值微分：用微小的差分求导数
def numerical_diff(f, x):
    h = 1e-4 # 这个值不能太小，否则会导致舍入误差，如np.float(1e-50)为0.0
    # return f(x + h) - f(x) / h # 前向差分
    return (f(x + h) - f(x - h)) / (2 * h) # 中心差分，比前向差分的误差更小

# 测试函数
def function_1(x):
    return 0.01 * x**2 + 0.1 * x

# 切线图
def tangent_line(f, x):
    a = numerical_diff(f, x)
    b = f(x) - a * x
    return lambda t: a * t + b # 返回匿名函数：f(t) = a * t + b

def test():
    print("-----数值微分-----")
    print("x=5:")
    print(numerical_diff(function_1, 5))
    print("x=10:")
    print(numerical_diff(function_1, 10))

    print("-----绘制函数的图像和切线-----")
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)

    tl1 = tangent_line(function_1, 5)
    y1 = tl1(x)

    tl2 = tangent_line(function_1, 10)
    y2 = tl2(x)
    
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y, label="f(x)")
    plt.plot(x, y1, label="tangent line at x=5", linestyle="--")
    plt.plot(x, y2, label="tangent line at x=10", linestyle=":")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test()