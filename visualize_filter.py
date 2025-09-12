import numpy as np
import matplotlib.pyplot as plt
from convolutional_neural_network import SimpleConvNet, Convolution
from matplotlib.image import imread

def filter_show(filters, title, nx=8):
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx)) # nx：每一行显示几个图，ny：显示多少行

    fig = plt.figure()

    # 调整子图之间、子图与画布边缘的间距
    # left, right, bottom, top - 子图的左右上下边界
    # hspace - 子图之间的垂直间距
    # wspace - 子图之间的水平间距
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        # 把画布分成ny行nx列的网格，在(i+1)个位置上添加子图
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])

        # filters[i, 0]是二维数组(FH, FW)
        # imshow()用于把二维数组显示成灰度图
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

# 显示学习前/后的滤波器
filter_show(network.params['W1'], title="Before Training")

network.load_params("params.pkl")
filter_show(network.params['W1'], title="After Training") # 学习前的滤波器无规律，学习后有规律 - 从白到黑渐变、含有块状区域blob

# 在图像上应用学习后的滤波器
img = imread("./dataset/lena_gray.png")
img = img.reshape(1, 1, *img.shape)

fig = plt.figure()
for i in range(16):
    W = network.params['W1'][i]
    b = network.params['b1'][i]

    W = W.reshape(1, *W.shape)
    b = b.reshape(1, *b.shape)
    
    conv_layer = Convolution(W, b)
    out = conv_layer.forward(img)
    out = out.reshape(out.shape[2], out.shape[3])

    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    ax.imshow(out, cmap=plt.cm.gray_r, interpolation='nearest')

plt.suptitle("Apply filters")
plt.show()