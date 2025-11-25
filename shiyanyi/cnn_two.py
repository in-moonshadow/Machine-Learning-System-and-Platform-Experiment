import numpy as np


class TwoLayerNeuralNetwork:
    """两层神经网络分类器"""

    def __init__(self, hidden_size=100, learning_rate=0.1, n_iterations=1000, reg_lambda=0.01):
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.reg_lambda = reg_lambda
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def relu(self, z):
        """ReLU激活函数"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """ReLU导数"""
        return (z > 0).astype(float)

    def softmax(self, z):
        """Softmax函数"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        """交叉熵损失"""
        n_samples = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        loss = -np.sum(y_true * np.log(y_pred)) / n_samples
        # L2正则化
        loss += 0.5 * self.reg_lambda * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return loss

    def fit(self, X, y):
        """训练模型"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # 初始化权重
        self.W1 = np.random.randn(n_features, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, n_classes) * 0.01
        self.b2 = np.zeros((1, n_classes))

        # One-hot编码
        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y] = 1

        # 训练
        for iteration in range(self.n_iterations):
            # 前向传播
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self.relu(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self.softmax(z2)

            # 计算损失
            loss = self.cross_entropy_loss(a2, y_one_hot)

            # 反向传播
            dz2 = a2 - y_one_hot
            dW2 = np.dot(a1.T, dz2) / n_samples + self.reg_lambda * self.W2
            db2 = np.sum(dz2, axis=0, keepdims=True) / n_samples

            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self.relu_derivative(z1)
            dW1 = np.dot(X.T, dz1) / n_samples + self.reg_lambda * self.W1
            db1 = np.sum(dz1, axis=0, keepdims=True) / n_samples

            # 更新参数
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.4f}")

    def predict(self, X):
        """预测"""
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.softmax(z2)
        return np.argmax(a2, axis=1)

    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
