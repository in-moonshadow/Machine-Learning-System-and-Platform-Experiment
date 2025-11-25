import numpy as np


class SoftmaxClassifier:
    """Softmax分类器"""

    def __init__(self, learning_rate=0.1, n_iterations=1000, reg_lambda=0.01):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.reg_lambda = reg_lambda
        self.W = None
        self.b = None

    def softmax(self, z):
        """Softmax函数"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        """交叉熵损失"""
        n_samples = y_true.shape[0]
        # 避免log(0)
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        loss = -np.sum(y_true * np.log(y_pred)) / n_samples
        # 添加L2正则化
        loss += 0.5 * self.reg_lambda * np.sum(self.W ** 2)
        return loss

    def fit(self, X, y):
        """训练模型"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # 初始化权重和偏置
        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros((1, n_classes))

        # 将标签转换为one-hot编码
        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y] = 1

        # 梯度下降
        for iteration in range(self.n_iterations):
            # 前向传播
            scores = np.dot(X, self.W) + self.b
            probs = self.softmax(scores)

            # 计算损失
            loss = self.cross_entropy_loss(probs, y_one_hot)

            # 反向传播
            dscores = probs - y_one_hot
            dW = np.dot(X.T, dscores) / n_samples + self.reg_lambda * self.W
            db = np.sum(dscores, axis=0, keepdims=True) / n_samples

            # 更新参数
            self.W -= self.lr * dW
            self.b -= self.lr * db

            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.4f}")

    def predict(self, X):
        """预测"""
        scores = np.dot(X, self.W) + self.b
        probs = self.softmax(scores)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
