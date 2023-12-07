import numpy as np
from numpy import newaxis


data = np.arange(2, 31, 2).reshape(3, 5)
print(data)
print(data.shape)
print(data.ndim)
print(data.size)
print(data.dtype.name)

A = np.array([(1, 2),
              (0, 1)])
B = np.array([(1, 2),
              (3, 4)])
print(A * B)  # 对应位置相乘
print(np.dot(A, B))  # 矩阵乘法

data_1 = np.random.random((2, 3))
print(data_1)
print(data_1.mean())  # 取所有元素平均数
print(data_1.sum())  # 对所有元素求和
print(data_1.max())  # 最大元素
print(data_1.min())  # 最小元素

b = np.arange(12).reshape(3, 4)
print(b)
print(b.sum(axis=0))  # 对每一列（0维）求和
print(b.sum(axis=1))  # 对每一行（1维）求和
print(b.min(axis=1))  # 求每一行最小值
print(b.cumsum(axis=1))  # 对每一行（1维)的元素累加求和

# 广播函数：对数组中的每个元素进行计算，返回计算后的数组。
B = np.arange(3)
print(B)
print(np.exp(B))
print(np.sqrt(B))
C = np.array([2, -1, 4])
print(np.add(B, C))
print(B + C)


# 多维数组可以在每一个维度有一个索引，这些索引构成元组来进行访问。
def f(x, y):
    return 10 * x + y


b = np.fromfunction(f, (5, 4), dtype=int)
# 以f为函数，输入index，输出该index下的函数值，生成（5,4）形状的矩阵
print(b)
print(b[2, 3])  # 取某一位置的值
print(b[0:5, 1])  # 切片
print(b[:, 1])
print(b[1:3, :])

# 对多维数组的迭代是在第一维进行迭代的。
for row in b:
    print(row)  # 输出每一行
# 如果需要遍历多维数组的所有元素，可以使用flat这个属性。
for element in b.flat:
    print(element)  # 输出每个元素

# 更改数组形状
a = np.floor(10 * np.random.random((3, 4)))  # np.floor返回不大于输入的最大整数
print(a)
print(a.ravel())  # 返回铺平后的数组
print(a.reshape(6, 2))  # 按照指定的形状更改
print(a.T)  # 返回转置矩阵

# 沿不同方向堆砌矩阵
a = np.floor(10 * np.random.random((2, 2)))
print(a)
b = np.floor(10 * np.random.random((2, 2)))
print(b)
print(np.vstack((a, b)))  # 垂直方向堆砌
print(np.hstack((a, b)))  # 水平方向堆砌
print(a[:, newaxis])  # np.newaxis:增加一个轴

# 使用hsplit，vsplit可以对数组按照水平方向和垂直方向进行划分。

a = np.floor(10 * np.random.random((2, 12)))
print(a)
print(np.hsplit(a, 3))
print(np.hsplit(a, (1, 2, 3)))  # 在第一列，第二列，第三列进行划分

# 简单的赋值不会复制数组的数据，而是创造了一个索引b，b也指向同样的数组。
# 这就意味着，无论对a还是b操作，都是在对同一个数组操作。
a = np.arange(12)
b = a
print(b is a)
b.shape = 3, 4
print(a.shape)
print(a)
print(b)

# 不同的数组可以使用同一份数据，view函数在同一份数据上创建了新的数组对象。
c = a.view()
print(c is a)
print(c.base is a)  # c是a的数据的视图
print(c.flags.owndata)
c.shape = 6, 2
print(a.shape)  # a的形状没有改变
c[4, 1] = 1234  # a的数据改变了
print(a)

# copy函数实现了对数据和数组的完全复制。
# 这种情况下，a,d是完全不同的两个数组。
d = a.copy()
print(d is a)
print(d.base is a)
d[0, 0] = 9999
print(a)

# 线性代数简单的数组操作
a = np.array([[1.0, 2.0], [3.0, 4.0]])
print(a)
a.transpose()
np.linalg.inv(a)
u = np.eye(2)  # unit 2x2 matrix; "eye" represents "I"
j = np.array([[0.0, -1.0], [1.0, 0.0]])
np.dot(j, j)  # 点积
np.trace(u)  # 矩阵的迹
y = np.array([[5.], [7.]])
print(np.linalg.solve(a, y))  # 解线性方程组
print(np.linalg.eig(j))  # 计算特征值


