import numpy as np
import pickle
from collections import OrderedDict
from layers import *
from numerical_gradient import numerical_gradient


# image to column函数
# 参数：
#    input_data: 4维数组(数据量，通道，高，宽)
#    filter_h：滤波器的高
#    filter_w：滤波器的宽
#    stride: 步幅
#    pad：填充
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape

    # 输出的高和宽
    out_h = (H + 2*pad - filter_h) // stride + 1 # //为地板除法，即进行除法后再向下取整
    out_w = (W + 2*pad - filter_w) // stride + 1

    # 给输入数据进行填充（padding）
    # 具体地：
    #    第0维（N）：前后都不填充
    #    第1维（C）：前后都不填充
    #    第2维（H）：前后各填充pad个元素
    #    第3维（W）：前后各填充pad个元素
    #    填充模式constant：用常数0填充
    img = np.pad(input_data, [(0,0), (0,0), (pad,pad), (pad,pad)], mode='constant')

    # 生成6维全0数组
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 对于滤波器的每一个位置(y, x)，取出滤波器在图像上滑动时，对应这个位置的图像的位置
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    
    # 转换成2维数组(N * out_h * out_w, C * filter_h * filter_w)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


# column to image函数
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    
    return img[:, :, pad:H+pad, pad:W+pad]


def test_im2col():
    print("-----测试im2col函数-----")
    x1 = np.random.rand(1, 3, 7, 7)
    col1 = im2col(x1, 5, 5, stride=1, pad=0)
    print(col1.shape) # (9, 75)

    x2 = np.random.rand(10, 3, 7, 7)
    col2 = im2col(x2, 5, 5, stride=1, pad=0)
    print(col2.shape) # (90, 75)


# 卷积层
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W # 权重，即滤波器
        self.b = b
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

    
    def forward(self, x):
        FN, C, FH, FW = self.W.shape # 滤波器数量，通道数，滤波器高度，滤波器宽度
        N, C, H, W = x.shape # 数据量，通道数，数据高度，数据宽度

        # 输出大小
        out_h = int((H + 2*self.pad - FH) / self.stride + 1)
        out_w = int((W + 2*self.pad - FW) / self.stride + 1)

        # 展开数据成2维数组(N * out_h * out_w, C * FH * FW)
        col = im2col(x, FH, FW, stride=self.stride, pad=self.pad)

        # 展开滤波器成2维数组(C * FH * FW, FN)
        col_W = self.W.reshape(FN, -1).T

        # 计算输出数组，为2维数组(N * out_h * out_w, FN)
        out = np.dot(col, col_W) + self.b

        # 把输出数组转换成4维数组(N, FN, out_h, out_w)
        # transpose用于更改多维数组的轴的顺序，reshape后的数组为(N, out_h, out_w, FN)，transpose后调整为(N, FN, out_h, out_w)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out


    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
    

# 池化层
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None
    

    def forward(self, x):
        N, C, H, W = x.shape

        # 输出大小
        out_h = int((H - self.pool_h) / self.stride + 1)
        out_w = int((W - self.pool_w) / self.stride + 1)

        # 展开输入数据，即二维数组(N * C * out_h * out_w, pool_h * pool_w)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 按行求最大值
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)

        # 转换成输出数据，即4维数组(N, C, out_h, out_w)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out
    

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


# 手写数字识别的CNN
class SimpleConvNet:
    # 参数:
    #    input_dim: 输入数据的维度(通道，高，宽)
    #    conv_param: 卷积层的超参数
    #    hidden_size: 隐藏层的神经元数量
    #    output_size: 输出层的神经元数量
    #    weight_init_std: 初始化权重的标准差
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        # 卷积层参数
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']

        # 输入数据的大小（高和宽）
        input_size = input_dim[1]

        # 卷积层输出的大小（高和宽）
        conv_output_size = (input_size + 2*filter_pad - filter_size) / filter_stride + 1

        # 池化层输出的大小
        pool_output_size = int(filter_num * (conv_output_size/2)**2)

        # 初始化权重
        self.params = {}

        # 卷积层权重
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)

        # 隐藏层权重
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)

        # 输出层权重
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 生成所有的层
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], filter_stride, filter_pad)
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size : (i+1)*batch_size]
            tt = t[i*batch_size : (i+1)* batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        
        return acc / x.shape[0]
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 计算梯度
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads


    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])
        
        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
    

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        
        for key, val in params.items():
            self.params[key] = val
        
        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]


def gradient_check():
    print("-----gradient check-----")
    network = SimpleConvNet(input_dim=(1,10,10),
                            conv_param={'filter_num':10, 'filter_size':3, 'pad':0, 'stride':1},
                            hidden_size=10, output_size=10, weight_init_std=0.01)
    
    x = np.random.rand(100).reshape((1, 1, 10, 10))
    t = np.array([1]).reshape(1, 1)

    grad_num = network.numerical_gradient(x, t)
    grad = network.gradient(x, t)

    for key, val  in grad_num.items():
        print(key, np.abs(grad_num[key] - grad[key]).mean())


def test_save_load_params():
    print("------test save & load params-----")
    network = SimpleConvNet(input_dim=(1,10,10),
                            conv_param={'filter_num':10, 'filter_size':3, 'pad':0, 'stride':1},
                            hidden_size=10, output_size=10, weight_init_std=0.01)
    print("Network initial params:")
    print(network.params['W2'])

    network.save_params()
    network.load_params()
    print("Loaded params:")
    print(network.params['W2'])


if __name__ == '__main__':
    test_im2col()
    gradient_check()
    test_save_load_params()