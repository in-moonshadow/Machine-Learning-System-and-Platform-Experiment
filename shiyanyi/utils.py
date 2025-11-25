import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    """加载并预处理数据"""
    # 使用sklearn的手写数字数据集作为示例
    digits = load_digits()
    X = digits.data
    y = digits.target

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练集、验证集和测试集
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_accuracy(y_true, y_pred):
    """计算准确率"""
    return np.mean(y_true == y_pred)
