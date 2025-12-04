import matplotlib.pyplot as plt
import numpy as np
from shiyanyi.cnn_two import TwoLayerNeuralNetwork
from shiyanyi.knn_class import KNNClassifier
from shiyanyi.softmax_class import SoftmaxClassifier
from shiyanyi.svm_class import SVMClassifier
from utils import load_data

# ===== 中文字体配置 =====
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def visualize_test_predictions(model, X_test, y_test, model_name, num_samples=10):
    """
    针对手写数字数据集的可视化
    优化：自动检测图像尺寸 (8x8 或 28x28)
    """
    try:
        # 随机选择测试样本
        indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
        sample_images = X_test[indices]
        sample_labels = y_test[indices]

        # 预测
        if hasattr(model, 'predict'):
            predictions = model.predict(sample_images)
        else:
            predictions = model.forward(sample_images).argmax(axis=1)

        # === 优化点：动态计算图像尺寸 ===
        feature_dim = X_test.shape[1]
        img_size = int(np.sqrt(feature_dim))  # 64->8, 784->28

        # 创建可视化图
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        accuracy_text = f'{model.score(X_test, y_test):.4f}'
        fig.suptitle(f'{model_name} - 预测抽样 (Acc: {accuracy_text})', fontsize=14, fontweight='bold')

        for i, ax in enumerate(axes.flat):
            if i < len(sample_images):
                image = sample_images[i].reshape(img_size, img_size)
                ax.imshow(image, cmap='gray')

                # 预测正确绿色，错误红色
                color = '#2ecc71' if sample_labels[i] == predictions[i] else '#e74c3c'
                ax.set_title(f'T:{sample_labels[i]} | P:{predictions[i]}', color=color, fontweight='bold')
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{model_name.replace(" ", "_")}_test_predictions.png', dpi=300, bbox_inches='tight')
        return True

    except Exception as e:
        print(f"可视化错误: {e}")
        return False


def plot_model_comparison(results):
    """
    绘制模型性能对比柱状图
    优化点：动态调整Y轴范围，突出模型间的差异
    """
    models = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(10, 7))

    # 使用更有区分度的配色
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6']
    bars = plt.bar(models, accuracies, color=colors[:len(models)], edgecolor='black', alpha=0.8, width=0.6)

    # 在柱子上方显示准确率数值
    min_acc = min(accuracies)
    max_acc = max(accuracies)

    # === 核心优化：动态设置 Y 轴范围 ===
    # 将 Y 轴下限设置为 (最低准确率 - 缓冲值)，使差异看起来更明显
    # 如果准确率都很高(>0.9)，则从 0.9 或更接近最小值的地方开始
    buffer = (max_acc - min_acc) * 0.5 if max_acc != min_acc else 0.01
    ylim_bottom = max(0, min_acc - 0.02)  # 至少保留 2% 的空间
    ylim_top = min(1.0, max_acc + 0.01)  # 上限稍微高一点

    plt.ylim(ylim_bottom, ylim_top)

    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.0005,
                 f'{accuracy:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.xlabel('模型', fontsize=12, fontweight='bold')
    plt.ylabel('测试集准确率', fontsize=12, fontweight='bold')
    plt.title('MNIST手写数字分类 - 模型性能对比 (局部放大)', fontsize=15, fontweight='bold')
    plt.xticks(rotation=0)  # 模型名字较短时不需要旋转
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 添加最佳准确率参考线
    plt.axhline(y=max_acc, color='red', linestyle='--', alpha=0.6, linewidth=1.5,
                label=f'最佳准确率: {max_acc:.4f}')

    plt.legend(loc='lower right')  # 图例放在右下角避免遮挡

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')


def plot_confusion_matrix_for_best_model(model, X_test, y_test, model_name):
    """为最佳模型绘制混淆矩阵"""
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        elif hasattr(model, 'forward'):
            y_pred = model.forward(X_test).argmax(axis=1)
        else:
            return

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[str(i) for i in range(10)],
                    yticklabels=[str(i) for i in range(10)])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'{model_name} - 混淆矩阵')

        plt.tight_layout()
        plt.savefig(f'{model_name.replace(" ", "_")}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        return True
    except Exception as e:
        print(f"生成混淆矩阵失败: {e}")
        return False


def main():
    print("=" * 60)
    print("MNIST手写数字分类实验 (优化绘图版)")
    print("=" * 60)

    # 1. 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # 自动检测数据集类型
    feat_dim = X_train.shape[1]
    if feat_dim == 784:
        print("检测到数据格式: MNIST (28x28)")
    elif feat_dim == 64:
        print("检测到数据格式: Digits (8x8)")
    else:
        print(f"检测到未知数据维度: {feat_dim}")

    results = {}
    model_instances = {}

    # ==================== 模型训练部分 (保持不变) ====================

    # 1. KNN
    print("\n[KNN] 正在训练...")
    # 为了演示，直接使用一个较优参数，您可以换回原来的超参数搜索代码
    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)
    acc_knn = knn.score(X_test, y_test)
    results['KNN'] = acc_knn
    model_instances['KNN'] = knn
    print(f"KNN Acc: {acc_knn:.4f}")

    # 2. SVM
    print("\n[SVM] 正在训练...")
    svm = SVMClassifier(learning_rate=0.001, lambda_param=0.01, n_iterations=200)  # 减少迭代加速演示
    svm.fit(X_train, y_train)
    acc_svm = svm.score(X_test, y_test)
    results['SVM'] = acc_svm
    model_instances['SVM'] = svm
    print(f"SVM Acc: {acc_svm:.4f}")

    # 3. Softmax
    print("\n[Softmax] 正在训练...")
    softmax = SoftmaxClassifier(learning_rate=0.5, n_iterations=200, reg_lambda=0.001)
    softmax.fit(X_train, y_train)
    acc_soft = softmax.score(X_test, y_test)
    results['Softmax'] = acc_soft
    model_instances['Softmax'] = softmax
    print(f"Softmax Acc: {acc_soft:.4f}")

    # 4. Neural Network
    print("\n[Neural Network] 正在训练...")
    nn = TwoLayerNeuralNetwork(hidden_size=64, learning_rate=0.5, n_iterations=200)
    nn.fit(X_train, y_train)
    acc_nn = nn.score(X_test, y_test)
    results['NN'] = acc_nn
    model_instances['NN'] = nn
    print(f"NN Acc: {acc_nn:.4f}")

    # ==================== 可视化部分 ====================
    print("\n" + "=" * 60)
    print("生成优化后的可视化图表...")

    # 1. 绘制对比图 (核心修改点)
    plot_model_comparison(results)
    print("✓ 对比图已保存 (Y轴已缩放): model_comparison.png")

    # 2. 预测采样图 (修复了尺寸问题)
    for name, model in model_instances.items():
        visualize_test_predictions(model, X_test, y_test, name)
    print("✓ 预测采样图已保存")

    # 3. 最佳模型混淆矩阵
    best_name = max(results, key=results.get)
    plot_confusion_matrix_for_best_model(model_instances[best_name], X_test, y_test, best_name)
    print(f"✓ 最佳模型 ({best_name}) 混淆矩阵已保存")


if __name__ == "__main__":
    main()
