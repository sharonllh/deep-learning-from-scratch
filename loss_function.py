import numpy as np

# 均方误差：一个训练数据
# def mean_squared_error(y, t):
#     return 0.5 * np.sum((y - t) ** 2)

# 交叉熵误差：一个训练数据
# def cross_entropy_error(y, t):
#     delta = 1e-7
#     return -np.sum(t * np.log(y + delta)) # 用delta防止出现log0 - 结果为负无限大-inf

# 均方误差：多个训练数据
def mean_squared_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size) # 把一维数组变成1行y.size列的二维数组
        t = t.reshape(1, t.size)
    
    batch_size = y.shape[0] # 训练数据的个数
    return 0.5 * np.sum((y - t) ** 2) / batch_size

# 交叉熵误差：多个训练数据
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size) # 把一维数组变成1行y.size列的二维数组
        t = t.reshape(1, t.size)
    
    batch_size = y.shape[0] # 训练数据的个数
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size # 用delta防止出现log0 - 结果为负无限大-inf

# 测试损失函数
def test():
    print("-----测试损失函数------")
    print("均方误差：")
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y1 = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    print(mean_squared_error(y1, t))

    y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
    print(mean_squared_error(y2, t))

    print("交叉熵误差：")
    print(cross_entropy_error(y1, t))
    print(cross_entropy_error(y2, t))

    print("-----测试多个训练数据的损失函数-----")
    y3 = np.array([y1, y2]) # 2*10的二维数组
    t3 = np.array([t, t])

    print("均方误差：")
    print(mean_squared_error(y3, t3))

    print("交叉熵误差：")
    print(cross_entropy_error(y3, t3))


if __name__ == '__main__':
    test()