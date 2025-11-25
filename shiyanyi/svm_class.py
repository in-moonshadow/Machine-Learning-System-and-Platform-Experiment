import numpy as np


class SVMClassifier:
    """支持向量机分类器（使用梯度下降的简化版本）"""

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = {}
        self.bias = {}

    def fit(self, X, y):
        """训练模型（One-vs-Rest策略）"""
        self.classes = np.unique(y)
        n_features = X.shape[1]

        # 为每个类别训练一个二分类器
        for cls in self.classes:
            # 将当前类别标记为1，其他为-1
            y_binary = np.where(y == cls, 1, -1)

            # 初始化权重和偏置
            w = np.zeros(n_features)
            b = 0

            # 梯度下降
            for _ in range(self.n_iterations):
                for idx, x_i in enumerate(X):
                    condition = y_binary[idx] * (np.dot(x_i, w) - b) >= 1

                    if condition:
                        w -= self.lr * (2 * self.lambda_param * w)
                    else:
                        w -= self.lr * (2 * self.lambda_param * w - np.dot(x_i, y_binary[idx]))
                        b -= self.lr * y_binary[idx]

            self.weights[cls] = w
            self.bias[cls] = b

    def predict(self, X):
        """预测"""
        predictions = []

        for x in X:
            scores = []
            for cls in self.classes:
                score = np.dot(x, self.weights[cls]) - self.bias[cls]
                scores.append(score)

            # 选择得分最高的类别
            prediction = self.classes[np.argmax(scores)]
            predictions.append(prediction)

        return np.array(predictions)

    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
