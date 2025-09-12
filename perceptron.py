import numpy as np

# 与门
# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = x1 * w1 + x2 * w2
#     if tmp > theta:
#         return 1
#     else:
#         return 0
    
# 与门
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    val = np.sum(x * w) + b
    if val > 0:
        return 1
    else:
        return 0

print("---测试与门---")
print(AND(0, 0)) # 0
print(AND(0, 1)) # 0
print(AND(1, 0)) # 0
print(AND(1, 1)) # 1

# 与非门
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    val = np.sum(x * w) + b
    if val > 0:
        return 1
    else:
        return 0

print("---测试与非门---")
print(NAND(0, 0)) # 1
print(NAND(0, 1)) # 1
print(NAND(1, 0)) # 1
print(NAND(1, 1)) # 0

# 或门
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.3
    val = np.sum(x * w) + b
    if val > 0:
        return 1
    else:
        return 0

print("---测试或门---")
print(OR(0, 0)) # 0
print(OR(0, 1)) # 1
print(OR(1, 0)) # 1
print(OR(1, 1)) # 1

# 异或门：由与门、与非门、或门组合实现
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print("---测试异或门---")
print(XOR(0, 0)) # 0
print(XOR(0, 1)) # 1
print(XOR(1, 0)) # 1
print(XOR(1, 1)) # 0