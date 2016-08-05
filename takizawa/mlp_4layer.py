#python 3.4.4
from scipy import misc
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

class Layer:
    def __init__(self,W,b,f):
        self._W = W
        self._b = b
        self._f = f

    def propagate_forward(self,x):
        return self._f(self._W.dot(x) + self._b)

def sigmoid(s):
    return 1/(1 +  np.exp(-s))

def d_sigmoid(y):
    return y * (1- y)

def se(t,y):
    return ((t-y).T.dot((t-y))).flatten()[0]/2

def d_se(t, y):
    return -(t-y)

def ma(history,n):
    return np.array([0,] * (n - 1) + [np.average(history[i - n: i]) for i in range(n, len(history) + 1)])


if __name__ == '__main__':
    from mnist import MNIST

    # Load MNIST dataset
    mndata = MNIST('./mnist')
    train_img,train_label = mndata.load_training()
    train_img = np.array(train_img, dtype=float)/255.0
    train_label = np.array(train_label, dtype = float)

    #Input vector (Layer 0)
    n_output_0 = len(train_img[0])

    #Middle layer (Layer 1)
    n_output_1 = 200
    W1 = np.random.randn(n_output_1, n_output_0)
    b1 = np.random.randn(n_output_1, 1)
    layer1 = Layer(W1, b1, sigmoid)

    #Middle layer (Layer 2)
    n_output_2 = 100
    W2 = np.random.randn(n_output_2, n_output_1)
    b2 = np.random.randn(n_output_2, 1)
    layer2 = Layer(W2, b2, sigmoid)

    # Output layer (Layer 3)
    n_output_3 = 10
    W3 = np.random.randn(n_output_3, n_output_2)
    b3 = np.random.randn(n_output_3, 1)
    layer3 = Layer(W3, b3, sigmoid)



    # FP BP and leaning
    epsilon = 0.15
    n_training_data = 1000
    se_history = []
    y1_history = []
    y2_history = []
    y3_history = []
    W1_history = []
    W2_history = []
    W3_history = []
    cpr_history = []

    for loop in range(100):
        for i in tqdm(range(n_training_data)):
            # Store W1 and W2 history
            W1_history.append(np.linalg.norm(layer1._W))
            W2_history.append(np.linalg.norm(layer2._W))
            W3_history.append(np.linalg.norm(layer3._W))

            # FP
            x = train_img[i].reshape(len(train_img[i]), 1)
            y1 = layer1.propagate_forward(x)
            y2 = layer2.propagate_forward(y1)
            y3 = layer3.propagate_forward(y2)

            # Store y1 and y2
            y1_history.append(y1)
            y2_history.append(y2)
            y3_history.append(y3)

            # Training datum
            t = np.zeros(shape=(10,1))
            t[train_label[i], 0] = 1.0

            # Calculate and store SE
            se_history.append(se(t,y3))

            # BP
            delta3 = d_se(t, y3) * d_sigmoid(y3)
            delta2 = layer3._W.T.dot(delta3) * d_sigmoid(y2)
            delta1 = layer2._W.T.dot(delta2) * d_sigmoid(y1)

            # Learning
            Delta_W3 =  delta3.dot(y2.T)
            layer3._W -= epsilon * Delta_W3
            layer3._b -= epsilon * delta3

            Delta_W2 = delta2.dot(y1.T)
            layer2._W -= epsilon * Delta_W2
            layer2._b -= epsilon * delta2

            Delta_W1 = delta1.dot(x.T)
            layer1._W -= epsilon * Delta_W1
            layer1._b -= epsilon * delta1

        #FP to evaluate correct prediction rate
        n_correct_prediction = 0
        n_prediction = 0
        for _i in np.random.choice(np.arange(n_training_data, train_img.shape[0]), 100):
            _x =train_img[_i].reshape(len(train_img[_i]), 1)
            _y1 = layer1.propagate_forward(_x)
            _y2 = layer2.propagate_forward(_y1)
            _y3 = layer3.propagate_forward(_y2)

            n_prediction += 1
            if train_label[_i] == np.argmax(_y3):
                n_correct_prediction += 1
        cpr_history.append(n_correct_prediction/n_prediction)

    #Draw W1
    plt.figure()
    plt.title('W1 history')
    plt.plot(range(len(W1_history)),W1_history)

    #Draw W2
    plt.figure()
    plt.title('W2 history')
    plt.plot(range(len(W2_history)),W2_history)

    #Draw W3
    plt.figure()
    plt.title('W3 history')
    plt.plot(range(len(W3_history)),W3_history)

    #Draw SE history and its moving average
    plt.figure()
    plt.title('SE history')
    plt.plot(range(len(se_history)),se_history,color = 'green')
    plt.plot(range(len(se_history)),ma(se_history,100),color = 'red')

    #Draw CPR history
    plt.figure()
    plt.title('CPR')
    plt.plot(range(len(cpr_history)),cpr_history)

    plt.show()
