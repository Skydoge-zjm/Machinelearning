import math


def activate(x, g, delta=None, alpha=None, beta=None):
    y = 0
    if g == 'relu':
        y = relu(x)
    elif g == 'linear':
        y = linear(x)
    elif g == 'sigmoid':
        y = sigmoid(x)
    elif g == 'softmax':
        y = softmax(x)
    elif g == 'leaky_relu':
        y = leaky_relu(x, delta)
    elif g == 'elu':
        y = elu(x, delta)
    elif g == 'selu':
        y = selu(x, delta, alpha)
    elif g == 'swish':
        y = swish(x, beta)
    else:
        print("ERROR: No activation of '" + g + "'.")
    return y


def relu(x):
    return max(0, x)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def linear(x):
    return x


def softmax(x: list):
    n = len(x)
    y = sum(x)
    p = []
    for i in range(n):
        p.append(x[i] / y)
    return p


def tanh(x):
    return 2 / (1+math.exp(-2*x)) - 1


def leaky_relu(x, delta):
    if x >= 0:
        return x
    else:
        return delta * x


def elu(x, delta):
    if x >= 0:
        return x
    else:
        return delta * (math.exp(x)-1)


def selu(x, delta, alpha):
    if x >= 0:
        return alpha * x
    else:
        return alpha * delta * (math.exp(x)-1)


def swish(x, beta):
    return x * sigmoid(beta * x)
