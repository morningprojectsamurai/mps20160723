import numpy as np

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
    return 1 if s > 0 else 0

d_relu = np.vectorize(d_relu)

def se(t, y):
    '''
    損失関数
    '''
    return ((t - y).T @ (t - y)).flatten()[0] / 2

def d_se(t, y):
    '''
    損失関数の微分
    '''
    return -(t - y)

def ma(history, n):
    '''
    移動平均
    '''
    return np.array([0, ] * (n - 1) + \
            [np.average(history[(i - n): i]) for i in range(n, len(history) + 1)])

class Layer:
    '''
    ニューラルネットワークのレイヤークラス
    '''
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
    X_train = (train.drop(["label"], axis=1).values).astype(np.float) / 255

    # 教師データのバイナライズ
    binarizer = LabelBinarizer()
    y_train = binarizer.fit_transform(train.label).astype(float)

    # 入力画像の次元を定義
    n_output_0 = X_train.shape[1]

    # layer 1
    # - ユニット数: 200
    n_output_1 = 200
    W1 = np.random.randn(n_output_1, n_output_0)
    b1 = np.random.randn(n_output_1, 1)
    # layer1 = Layer(W1, b1, relu)
    layer1 = Layer(W1, b1, sigmoid)

    # output layer
    # - 出力層が10ユニット(10クラス分類のため)
    n_output_2 = 10
    W2 = np.random.randn(n_output_2, n_output_1)
    b2 = np.random.randn(n_output_2, 1)
    # layer2 = Layer(W2, b2, relu)
    layer2 = Layer(W2, b2, sigmoid)

    # learning rate
    epsilon = 0.15

    # batch size
    n_training_data = 1000

    # 学習の経過を保存するリスト
    se_history = []
    y1_history = []
    y2_history = []
    W1_history = []
    W2_history = []
    cpr_history = []

    for loop in range(400):

        # データの並び替え
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx, :]
        y_train = y_train[idx, :]

        for i in tqdm(range(j + n_training_data)):

            # 重み行列のノルムの計算し、履歴リストに追加
            W1_history.append(np.linalg.norm(layer1._W))
            W2_history.append(np.linalg.norm(layer2._W))

            # 潤伝搬計算
            x = X_train[i, :].reshape(n_output_0, 1)
            y1 = layer1.propagate_forward(x)
            y2 = layer2.propagate_forward(y1)

            # 各層の(活性化関数適用後の)出力を履歴に追加
            y1_history.append(y1)
            y2_history.append(y2)

            # ネットワークの出力
            t = y_train[i, :].reshape(n_output_2, 1)

            # 誤差を履歴リストに追加
            se_history.append(se(t, y2))

            # 誤差逆伝播
            delta2 = d_se(t, y2) * d_relu(y2)
            delta1 = layer2._W.T @ delta2 * d_relu(y1)
            # delta2 = d_se(t, y2) * d_sigmoid(y2)
            # delta1 = layer2._W.T @ delta2 * d_sigmoid(y1)

            # 第二層の重み行列について、偏微分*学習率を除算
            Delta_W2 = delta2 @ y1.T
            layer2._W -= epsilon * Delta_W2
            layer2._b -= epsilon * delta2

            # 第一層の重み行列について、偏微分*学習率を除算
            Delta_W1 = delta1 @ x.T
            layer1._W -= epsilon * Delta_W1
            layer1._b -= epsilon * delta1

        # 学習に使用していないデータから100個取り出し、Accuracyを計算する
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
plt.savefig("w1_history.png", transparent=True)

# draw W2
plt.figure()
plt.title("W2 history")
plt.plot(range(len(W2_history)), W2_history)
plt.savefig("w2_history.png", transparent=True)

#draw SE history and its moving average
plt.figure()
plt.title("SE History")
plt.plot(range(len(se_history)), se_history, color="#24d4c4")
plt.plot(range(len(se_history)), ma(se_history, 100), color="#ec2396")
plt.savefig("se_history.png", transparent=True)

# draw CPR history
plt.figure()
plt.title('CPR')
plt.plot(range(len(cpr_history)), cpr_history)
plt.savefig("cpr.png", transparent=True)
