import numpy as np


class KNNClassifier:
    """K近邻分类器"""

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """训练模型（存储训练数据）"""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """预测"""
        predictions = []
        for x in X:
            # 计算欧氏距离
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

            # 找到k个最近邻
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]

            # 投票决定类别
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            prediction = unique[np.argmax(counts)]
            predictions.append(prediction)

        return np.array(predictions)

    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
