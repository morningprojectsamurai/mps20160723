import numpy as np


def sgd(param, grad, lr=0.1):
    '''
    確率的最急降下法
    '''
    return param - lr * grad


def momentum_sgd(param, delta_param, grad, lr=0.1, momentum=0.9):
    '''
    モメンタム確率的最急降下法
    '''

    return param + momentum * delta_param - lr * grad
