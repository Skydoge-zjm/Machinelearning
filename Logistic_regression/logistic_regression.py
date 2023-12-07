import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# f_wb_x
def sigmoid(z):
    s = 1 / (1 + math.exp(-z))
    return s


# gradient decent
def gradient_decent(rows, cols, x, y, w, b):
    # initialization
    dj_db = 0
    dj_dw_i = [0 for j in range(cols)]
    dj_dw = [0 for j in range(cols)]

    # calculate w_j (j is the num of features)
    for j in range(cols):
        for i in range(rows):
            z_i = np.dot(x[i], w) + b
            f_wb_i = sigmoid(z_i)
            dj_dw_i[j] = (f_wb_i - y[i]) * x[i, j]
            dj_dw[j] = dj_dw[j] + dj_dw_i[j]
        dj_dw[j] = (1 / m) * dj_dw[j]

    # calculate b
    for i in range(rows):
        z_i = np.dot(x[i], w) + b
        f_wb_i = sigmoid(z_i)
        dj_db_i = f_wb_i - y[i]
        dj_db = dj_db + dj_db_i
    dj_db = (1 / m) * dj_db

    return dj_dw, dj_db


a = 0.1  # learning rate
t = 10000  # iterations

# input
n = int(input("Please input the number of features of each example:\n"))
m = 0
df = 0
if n == 2:
    m = 20
    df = pd.read_csv('ex1.csv')
if n == 3:
    m = 10
    df = pd.read_csv('ex2.csv')
# print(df)
data = np.array(df)
x = np.delete(data, n, axis=1)
# print(x)
y = np.delete(data, 0, axis=1)
for i in range(n-1):
    y = np.delete(y, 0, axis=1)
# print(y)


# normalization
max_f = [0, 0, 0]
min_f = [0, 0, 0]
for i in range(n):
    for j in range(m):
        if x[j][i] >= max_f[i]:
            max_f[i] = x[j][i]
        if x[j][i] <= min_f[i]:
            min_f[i] = x[j][i]
# print(max_f)
for i in range(n):
    for j in range(m):
        x[j][i] = (x[j][i]-min_f[i]) / (max_f[i]-min_f[i])
print(x)


# initialize w,b
temp = []
for i in range(n):
    temp.append(float(1))
w = np.array(temp)  # w has 1 row, n columns
print(w)
b = 0.0

# main body
for i in range(t):
    temp1 = gradient_decent(m, n, x, y, w, b)
    # temp1 is a tuple (a matrix, a number)
    for j in range(n):
        w[j] = w[j] - a * temp1[0][j]
    b = b - a * temp1[1]
    print(w, b)


# divide data, preparation for plt
f_1 = x[:, 0]
f_2 = x[:, 1]
temp_f_1_a = []
temp_f_1_b = []
temp_f_2_a = []
temp_f_2_b = []
v = y[:, 0]
temp_v_1 = []
temp_v_2 = []
for i in range(m):
    if v[i] == 1:
        temp_f_1_a.append(f_1[i])
        temp_f_2_a.append(f_2[i])
        temp_v_1.append(v[i])
    elif v[i] == 0:
        temp_f_1_b.append(f_1[i])
        temp_f_2_b.append(f_2[i])
        temp_v_2.append(v[i])

# plt
if n == 2:
    plt.figure(figsize=None, facecolor=None)
    plt.title('visualization')
    plt.xlabel('feature_1')
    plt.ylabel('feature_2')
    plt.grid()
    plt.scatter(temp_f_1_a, temp_f_2_a, c='red')
    plt.scatter(temp_f_1_b, temp_f_2_b, c='blue')
    axis_x = np.linspace(min(f_1)-abs(max(f_1)-min(f_1))*0.15, max(f_1)+abs(max(f_1)-min(f_1))*0.15, 500)
    axis_y = (-1)*w[0]/w[1] * axis_x - b[0]/w[1]
    plt.plot(axis_x, axis_y)
    plt.show()
# w0*x+w1*y=-b
# y = -w0/w1 * x - b/w1


# test
'''
2

20

1.32 4.46
1.76 7.24
2.22 3.48
3.34 1.22
5.14 4.00
4.00 6.00
5.30 5.04
6.00 4.34
0.76 1.28
5.70 6.82
7.64 4.80
5.64 0.42
2.02 2.56
3.76 7.74
7.88 3.40
2.04 0.78
4.16 0.24
5.82 5.64
8.00 2.00
6.72 3.10

0
1
0
0
1
1
1
1
0
1
1
0
0
1
1
0
0
1
1
1
'''


'''
3

10

1.82 2.22 2.16
-2.92 -0.81 4.28
4.73 -3.07 3.00
3.16 2.76 3.00
-1.48 3.75 4.22
4.27 4.08 4.73
5.94 3.39 4.28
5.46 1.32 3.28
-1.85 -3.80 5.28
2.48 -4.64 3.39

0
1
0
0
1
0
0
0
1
0

'''