import numpy as np
from utils.functions import softmax, sigmoid, d_sigmoid, d_se

class Layer(object):

    def __init__(self, w, b):
        self._w = w
        self._b = b

    def forward_propagation(self, x, activate_f):
        return activate_f(self._w @ x + self._b @ np.ones((1, x.shape[1])))

    def back_propagation(self, delta, data_in, data_out, grad_activate_f):
        return (self._w.T @ delta * grad_activate_f(data_out)) @ data_in.T

if __name__ == '__main__':

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import accuracy_score
    from tqdm import tqdm

    # データの読み込み / 整形
    train = pd.read_csv("train.csv")
    X = (train.drop(["label"], axis=1).values).astype(np.float) / 255

    # 教師データのバイナライズ
    binarizer = LabelBinarizer()
    y = binarizer.fit_transform(train.label).astype(np.float)

    # 入力層の次元
    n_units_0 = X.shape[1]

    # レイヤー1の定義
    n_units_1 = 200
    w1 = np.random.randn(n_units_1, n_units_0)
    b1 = np.random.randn(n_units_1, 1)
    layer1 = Layer(w1, b1)

    # レイヤー2の定義
    n_units_2 = y.shape[1]
    w2 = np.random.randn(n_units_2, n_units_1)
    b2 = np.random.randn(n_units_2, 1)
    layer2 = Layer(w2, b2)

    # パラメーターの指定
    batch_size = 1000
    test_size = 0.2
    lr = 0.1

    # 学習セットとテストセットに分割
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, stratify=train.label)

    # 学習過程を格納するリスト
    accuracy = []
    se = []
    w1_norm = []
    b1_norm = []
    w2_norm = []
    b2_norm = []

    for loop in tqdm(range(400)):
        # 並び替え
        idx = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[idx], y_train[idx]

        # ミニバッチ毎に学習
        for i in range(0, X_train.shape[0], batch_size):

            # ミニバッチデータの取得
            X_train_batch = X_train[i:(i + batch_size)]
            y_train_batch = y_train[i:(i + batch_size)]

            # FeedForward
            y1 = layer1.forward_propagation(X_train_batch.T, sigmoid)
            y2 = layer2.forward_propagation(y1, softmax)

            # BackPropagation
            delta2 = d_se(y_train_batch.T, y2)
            delta1 = (layer2._w.T @ delta2) * d_sigmoid(y1)
            grad_w2 = (delta2 @ y1.T)
            grad_b2 = delta2.mean(1).reshape(n_units_2, 1)
            grad_w1 = (delta1 @ X_train_batch)
            grad_b1 = delta1.mean(1).reshape(n_units_1, 1)

            # パラメーターの更新
            layer2._w -= lr * grad_w2 / X_train_batch.shape[0]
            layer2._b -= lr * grad_b2
            layer1._w -= lr * grad_w1 / X_train_batch.shape[0]
            layer1._b -= lr * grad_b1

            # テストセットで評価
            accuracy.append(accuracy_score(y2.T.argmax(1), y_train_batch.argmax(1)))
            w1_norm.append(np.linalg.norm(layer1._w))
            b1_norm.append(np.linalg.norm(layer1._b))
            w2_norm.append(np.linalg.norm(layer2._w))
            b2_norm.append(np.linalg.norm(layer2._b))

    # テストセットでの精度
    plt.plot(range(len(accuracy)), accuracy)
    plt.title("テストセットでのAccuracy")
    plt.xlabel("反復回数")
    plt.ylabel("精度")
    plt.savefig("accuracy_on_test.png", transparent=True)
