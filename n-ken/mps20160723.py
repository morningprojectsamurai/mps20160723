import numpy as np

# sigmoid関数の定義
def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def d_sigmoid(s):
    return y * (1 - y)

def se(t, y):
    return ((t - y).T @ (t - y)).flatten()[0] / 2

def d_se(t, y):
    return -(t - y)

def ma(history, n):
    return np.array([0, ] * (n - 1) + \
            [np.average(history[(i - n): i]) for i in range(n, len(history) + 1)])

x = np.random.randn(3, 1)
W = np.random.randn(1, 3)
b = np.random.randn(1, 1)
y = sigmoid(W @ x + b)

class Layer:
    def __init__(self, W, b, f):
        self._W = W
        self._b = b
        self._f = f

    def propagate_forward(self, x):
        return self._f(self._W @ x + self._b)

if __name__ == '__main__':

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelBinarizer
    from tqdm import tqdm

    # データの読み込み
    train = pd.read_csv("train.csv")
    X = (train.drop(["label"], axis=1).values).astype(np.float) / 255
    binarizer = LabelBinarizer()
    y = binarizer.fit_transform(train.label).astype(float)

    # データを訓練 / 検証セットに分割
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2)

    # input layer
    n_output_0 = X_train.shape[1]

    # layer 1
    n_output_1 = 200
    W1 = np.random.randn(n_output_1, n_output_0)
    b1 = np.random.randn(n_output_1, 1)
    layer1 = Layer(W1, b1, sigmoid)

    # output layer
    n_output_2 = 10
    W2 = np.random.randn(n_output_2, n_output_1)
    b2 = np.random.randn(n_output_2, 1)
    layer2 = Layer(W2, b2, sigmoid)

    # learning
    epsilon = 0.15
    # epsion = np.finfo(np.float).eps

    # batch size
    n_training_data = 1000 # X_train.shape[0]
    se_history = []
    y1_history = []
    y2_history = []
    W1_history = []
    W2_history = []
    cpr_history = []
    for loop in range(400):
        for i in tqdm(range(n_training_data)):

            # normalize and store w
            W1_history.append(np.linalg.norm(layer1._W))
            W2_history.append(np.linalg.norm(layer2._W))

            # forward propagation
            x = X_train[i, :].reshape(n_output_0, 1)
            y1 = layer1.propagate_forward(x)
            y2 = layer2.propagate_forward(y1)

            # store y1 and y2
            y1_history.append(y1)
            y2_history.append(y2)

            # t
            t = y_train[i, :].reshape(n_output_2, 1)

            se_history.append(se(t, y2))

            # back propagation
            delta2 = d_se(t, y2) * d_sigmoid(y2)
            delta1 = layer2._W.T @ delta2 * d_sigmoid(y1)

            # learning
            Delta_W2 = delta2 @ y1.T
            layer2._W -= epsilon * Delta_W2
            layer2._b -= epsilon * delta2

            Delta_W1 = delta1 @ x.T
            layer1._W -= epsilon * Delta_W1
            layer1._b -= epsilon * delta1

        n_correct_prediction = 0
        n_prediction = 0
        for _i in np.random.choice(np.arange(n_training_data, X_train.shape[0]), 100):
            _x = X_train[_i, :].reshape(n_output_0, 1)
            _y1 = layer1.propagate_forward(_x)
            _y2 = layer2.propagate_forward(_y1)

            n_prediction += 1
            if y_train[_i, :].argmax() == _y2.argmax():
                n_correct_prediction += 1
        cpr_history.append(n_correct_prediction / n_prediction)

# draw W1
plt.figure()
plt.title("W1 history")
plt.plot(range(len(W1_history)), W1_history)

# draw W2
plt.figure()
plt.title("W2 history")
plt.plot(range(len(W2_history)), W2_history)

#draw SE history and its moving average
plt.figure()
plt.title("SE History")
plt.plot(range(len(se_history)), se_history, color="#24d4c4")
plt.plot(range(len(se_history)), ma(se_history, 100), color="#ec2396")

# draw CPR history
plt.figure()
plt.title('CPR')
plt.plot(range(len(cpr_history)), cpr_history)
