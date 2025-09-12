import numpy as np

# 乘法层
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    # 正向传播
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    # 反向传播
    # dout: 从上游传来的导数
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

# 加法层
class AddLayer:
    def __init__(self):
        pass # 占位语句，啥也不干

    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout
        dy = dout
        return dx, dy

def test():
    print("-----购买2个苹果-----")
    apple = 100
    apple_num = 2
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    # 前向传播
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)

    print("前向传播:")
    print(price) # 220

    # 反向传播
    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    print("后向传播:")
    print(dapple, dapple_num, dtax) # 2.2 110 200

    print("-----购买2个苹果和3个橘子-----")
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()

    # 前向传播
    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(all_price, tax)
    print("前向传播:")
    print(price) # 715

    # 反向传播
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)
    print("反向传播:")
    print(dapple, dapple_num, dorange, dorange_num, dtax) # 2.2 110 3.3 165 650

if __name__ == '__main__':
    test()
