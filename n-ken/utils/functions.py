import numpy as np

def softmax(s):
    '''
    ソフトマックス関数
    '''
    return np.exp(s) / np.exp(s).sum(0)

def sigmoid(s):
    '''
    sigmoid関数
    '''
    return 1 / (1 + np.exp(-s))

def relu(s):
    '''
    relu
    '''
    return max(s, 0)

relu = np.vectorize(relu)

def d_sigmoid(s):
    '''
    sigmoid関数の微分
    '''
    return s * (1 - s)

def d_relu(s):
    '''
    reluの微分
    '''
    return f if s > 0 else 0

d_relu = np.vectorize(d_relu)

def se(t, y):
    '''
    損失関数
    '''
    return ((t - y).T @ (t - y)).flatten()[0] / 2

def se_seftmax(t, y):
    '''
    損失関数(ソフトマックス関数)
    '''
    return -(t * log(y)).sum()

def d_se(t, y):
    '''
    損失関数の微分(ソフトマックスと同じ結果になるので使い回しています)
    '''
    return -(t - y)

def ma(history, n):
    '''
    移動平均
    '''
    return np.array([0, ] * (n - 1) + \
            [np.average(history[(i - n): i]) for i in range(n, len(history) + 1)])
