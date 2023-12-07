import numpy as np


def calculate_loss(x, y, g, delta):
    """
    x:list，代表模型预测的一组数据
    y:list，代表真实样本对应的一组数据
    delta:参数
    """
    if g == 'MSE':
        return mse(x, y)
    elif g == 'L2':
        return l2(x, y)
    elif g == 'L1':
        return l1(x, y)
    elif g == 'Smooth_L1':
        return sl1(x, y)
    elif g == 'huber':
        return huber(x, y, delta)
    elif g == 'kl':
        return kl(y_true=y, y_pre=x)
    elif g == 'cross':
        return cross(y_true=y, y_pre=x)


def mse(x: list, y: list):
    assert len(x) == len(y)
    x = np.array(x)
    y = np.array(y)
    loss = np.sum(np.square(x - y)) / len(x)
    return loss


def l2(x: list, y: list):
    assert len(x) == len(y)
    x = np.array(x)
    y = np.array(y)
    loss = np.sqrt(np.sum(np.square(x - y)) / len(x))
    return loss


def l1(x: list, y: list):
    assert len(x) == len(y)
    x = np.array(x)
    y = np.array(y)
    loss = np.sum(np.abs(x - y)) / len(x)
    return loss


def sl1(x, y):
    assert len(x) == len(y)
    loss = 0
    for i_x, i_y in zip(x, y):
        tmp = abs(i_y - i_x)
        if tmp < 1:
            loss += 0.5 * (tmp ** 2)
        else:
            loss += tmp - 0.5
    return loss


def huber(x, y, delta):
    assert len(x) == len(y)
    loss = 0
    for i_x, i_y in zip(x, y):
        tmp = abs(i_y - i_x)
        if tmp <= delta:
            loss += 0.5 * (tmp ** 2)
        else:
            loss += tmp * delta - 0.5 * delta ** 2
    return loss


def kl(y_true: list, y_pre: list):
    """
    y_true,y_pre，分别是两个概率分布
    比如：px=[0.1,0.2,0.8]
        py=[0.3,0.3,0.4]
    """
    assert len(y_true) == len(y_pre)
    loss = 0
    for y, fx in zip(y_true, y_pre):
        loss += y * np.log(y / fx)
    return loss


def cross(y_true: list, y_pre: list):
    assert len(y_true) == len(y_pre)
    loss: int = 0
    for y, fx in zip(y_true, y_pre):
        loss += -y * np.log(fx)
    return loss


"""
a = [0.1, 0.2, 0.5, 0.2]
b = [0.1, 0.3, 0.4, 0.2]
k = calculate_loss(x=a, y=b, g='cross', delta=0)
print(k)
"""
