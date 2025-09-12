import numpy as np
import matplotlib.pylab as plt
from numerical_differentiation import numerical_diff

# 测试函数
def function_2(x):
    return np.sum(x**2)

# 计算梯度：函数f关于所有变量xi的偏导数组成的向量，x为一维数组
def numerical_gradient_1d(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # 生成和x的形状、数据类型都相同的全0数组

    # 计算关于xi的偏导数：固定其他变量，对xi进行数值差分
    for idx in range(x.size):
        xi = x[idx]

        # 计算f(x+h)
        x[idx] = xi + h
        fxh1 = f(x)

        # 计算f(x-h)
        x[idx] = xi - h
        fxh2 = f(x)

        # 计算关于xi的偏导数
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        # 还原x[idx]
        x[idx] = xi
    
    return grad

# 批量计算梯度，X为二维数组
def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        for idx, x in enumerate(X): # 迭代X，返回索引idx和元素x
            grad[idx] = numerical_gradient_1d(f, x)
        return grad

# 计算梯度，x为多维数组
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite']) # 迭代数组x，multi_index表示记录多维索引(i,j,...)，readwrite表示迭代元素时允许读写操作
    while not it.finished:
        idx = it.multi_index # 多维索引(i,j,...)
        #print(idx)
        tmp_val = x[idx] # 该索引处的值x[i][j]...
        #print(tmp_val)

        # 计算f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # 计算f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        # 计算偏导数
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        # 还原x[idx]
        x[idx] = tmp_val

        # 继续迭代下一个元素
        it.iternext()

    return grad


def test():
    print("-----测试偏导数-----")
    print("x0=3,x1=4时，关于x0的偏导数：")
    tmp1 = lambda x0: x0**2 + 4**2
    print(numerical_diff(tmp1, 3))

    print("x0=3,x1=4时，关于x1的偏导数：")
    tmp2 = lambda x1: 3**2 + x1**2
    print(numerical_diff(tmp2, 4))

    print("-----计算梯度-----")
    print("x0=3,x1=4:")
    print(numerical_gradient_1d(function_2, np.array([3.0, 4.0])))
    print(numerical_gradient(function_2, np.array([3.0, 4.0])))
    print("x0=0,x1=2:")
    print(numerical_gradient_1d(function_2, np.array([0.0, 2.0])))
    print(numerical_gradient(function_2, np.array([0.0, 2.0])))
    print("x0=3,x1=0:")
    print(numerical_gradient_1d(function_2, np.array([3.0, 0.0])))
    print(numerical_gradient(function_2, np.array([3.0, 0.0])))

    print("-----绘制梯度图-----")
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1) # 扩展成二维坐标, X=[[-2,-1.75,-1.5,...,2,2.25], [-2,-1.75,-1.5,...,2,2.25], ...], Y=[[-2,-2,-2,...-2],[-1.75,-1.75,-1.75,...],...]

    X = X.flatten() # 拉平为一维数组, X=[-2,-1.75,...,2,2.25,-2,-1.75,...2,2.25,...], Y=[-2,-2,...,-1.75,-1.75,...]
    Y = Y.flatten()

    grad = numerical_gradient_2d(function_2, np.array([X, Y]).T).T # .T为数组转置
    #print(grad)

    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666") # 绘制二维向量场，X,Y为箭头的起始坐标，-grad[0]和-grad[1]为箭头的分量
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.grid() # 显示网格线
    plt.show()
    
    print("-----计算多维数组的梯度-----")
    x = np.array([[0.47, 0.99, 0.84], [0.85, 0.03, 0.69]])
    f = lambda x: np.max(np.dot(np.array([0.6, 0.9]), x))
    grad = numerical_gradient(f, x)
    print(grad)


if __name__ == '__main__':
    test()