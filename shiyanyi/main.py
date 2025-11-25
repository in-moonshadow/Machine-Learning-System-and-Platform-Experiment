import matplotlib.pyplot as plt
import numpy as np
from shiyanyi.cnn_two import TwoLayerNeuralNetwork
from shiyanyi.knn_class import KNNClassifier
from shiyanyi.softmax_class import SoftmaxClassifier
from shiyanyi.svm_class import SVMClassifier
from utils import load_data, compute_accuracy
# ===== 中文字体配置（添加在所有导入之后，函数定义之前）=====
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def visualize_test_predictions(model, X_test, y_test, model_name, num_samples=10):
    """针对8×8手写数字数据集的可视化"""
    try:
        # 随机选择测试样本
        indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
        sample_images = X_test[indices]
        sample_labels = y_test[indices]

        # 预测
        if hasattr(model, 'predict'):
            predictions = model.predict(sample_images)
        else:
            # 其他预测方法
            predictions = model.forward(sample_images).argmax(axis=1)

        # 使用8×8图像尺寸（64维特征对应8×8）
        img_size = 8

        # 创建可视化图
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        fig.suptitle(f'{model_name} - 手写数字测试集预测结果 (准确率: {model.score(X_test, y_test):.4f})',
                     fontsize=14, fontweight='bold')

        for i, ax in enumerate(axes.flat):
            if i < len(sample_images):
                # 重塑为8×8图像
                image = sample_images[i].reshape(img_size, img_size)

                # 显示图像
                ax.imshow(image, cmap='gray')

                # 设置标题
                color = 'green' if sample_labels[i] == predictions[i] else 'red'
                ax.set_title(f'True: {sample_labels[i]} | Pred: {predictions[i]}', color=color)
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{model_name.replace(" ", "_")}_test_predictions.png', dpi=300, bbox_inches='tight')
        return True

    except Exception as e:
        print(f"可视化错误: {e}")
        return False

def plot_model_comparison(results):
    """绘制模型性能对比柱状图"""
    models = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(12, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A']
    bars = plt.bar(models, accuracies, color=colors[:len(models)], edgecolor='black', alpha=0.7)

    # 在柱子上方显示准确率数值
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{accuracy:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 设置图表属性
    plt.xlabel('模型', fontsize=14, fontweight='bold')
    plt.ylabel('测试集准确率', fontsize=14, fontweight='bold')
    plt.title('MNIST手写数字分类 - 模型性能对比', fontsize=16, fontweight='bold')
    plt.ylim(0, min(1.0, max(accuracies) + 0.1))
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    # 添加最佳准确率参考线
    best_acc = max(accuracies)
    plt.axhline(y=best_acc, color='red', linestyle='--', alpha=0.7,
                label=f'最佳准确率: {best_acc:.4f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')




def plot_confusion_matrix_for_best_model(model, X_test, y_test, model_name):
    """为最佳模型绘制混淆矩阵"""
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        # 预测
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        elif hasattr(model, 'forward'):
            y_pred = model.forward(X_test).argmax(axis=1)
        else:
            print(f"无法为{model_name}生成混淆矩阵")
            return

        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[str(i) for i in range(10)],
                    yticklabels=[str(i) for i in range(10)])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'{model_name} - 混淆矩阵 (MNIST数据集)')

        plt.tight_layout()
        plt.savefig(f'{model_name.replace(" ", "_")}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        return True
    except Exception as e:
        print(f"生成混淆矩阵失败: {e}")
        return False


def main():
    print("=" * 60)
    print("MNIST手写数字分类实验")
    print("=" * 60)

    # 加载数据
    print("\n1. 加载MNIST数据...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"测试集大小: {X_test.shape}")

    # 显示数据特征信息
    print(f"每个样本特征数: {X_train.shape[1]}")
    print(f"类别数: {len(np.unique(y_train))}")

    # 检查是否是MNIST数据
    if X_train.shape[1] == 784:  # MNIST是28x28=784
        print("检测到MNIST数据集 (28x28 灰度图像)")
    elif X_train.shape[1] == 64:  # 可能是8x8的digits数据集
        print("检测到8x8数字数据集")
    else:
        print(f"未知数据集，特征维度: {X_train.shape[1]}")

    results = {}
    model_instances = {}

    # ==================== KNN分类器 ====================
    print("\n" + "=" * 60)
    print("2. KNN分类器")
    print("=" * 60)

    best_k = 0
    best_val_acc = 0
    k_values = []
    val_accuracies = []

    # 超参数调整
    for k in [1, 3, 5, 7, 9]:
        knn = KNNClassifier(k=k)
        knn.fit(X_train, y_train)
        val_acc = knn.score(X_val, y_val)
        print(f"k={k}, 验证集准确率: {val_acc:.4f}")

        k_values.append(k)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_k = k

    # 使用最佳参数在测试集上评估
    print(f"\n最佳k值: {best_k}")
    knn = KNNClassifier(k=best_k)
    knn.fit(X_train, y_train)
    test_acc = knn.score(X_test, y_test)
    print(f"测试集准确率: {test_acc:.4f}")
    results['KNN'] = test_acc
    model_instances['KNN'] = knn

    # ==================== SVM分类器 ====================
    print("\n" + "=" * 60)
    print("3. SVM分类器")
    print("=" * 60)

    svm = SVMClassifier(learning_rate=0.001, lambda_param=0.01, n_iterations=500)
    svm.fit(X_train, y_train)

    train_acc = svm.score(X_train, y_train)
    val_acc = svm.score(X_val, y_val)
    test_acc = svm.score(X_test, y_test)

    print(f"训练集准确率: {train_acc:.4f}")
    print(f"验证集准确率: {val_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    results['SVM'] = test_acc
    model_instances['SVM'] = svm

    # ==================== Softmax分类器 ====================
    print("\n" + "=" * 60)
    print("4. Softmax分类器")
    print("=" * 60)

    softmax = SoftmaxClassifier(learning_rate=0.5, n_iterations=500, reg_lambda=0.001)
    softmax.fit(X_train, y_train)

    train_acc = softmax.score(X_train, y_train)
    val_acc = softmax.score(X_val, y_val)
    test_acc = softmax.score(X_test, y_test)

    print(f"\n训练集准确率: {train_acc:.4f}")
    print(f"验证集准确率: {val_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    results['Softmax'] = test_acc
    model_instances['Softmax'] = softmax

    # ==================== 两层神经网络 ====================
    print("\n" + "=" * 60)
    print("5. 两层神经网络")
    print("=" * 60)

    nn = TwoLayerNeuralNetwork(hidden_size=100, learning_rate=0.5,
                               n_iterations=500, reg_lambda=0.001)
    nn.fit(X_train, y_train)

    train_acc = nn.score(X_train, y_train)
    val_acc = nn.score(X_val, y_val)
    test_acc = nn.score(X_test, y_test)

    print(f"\n训练集准确率: {train_acc:.4f}")
    print(f"验证集准确率: {val_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    results['Neural Network'] = test_acc
    model_instances['Neural Network'] = nn

    # ==================== 可视化部分 ====================
    print("\n" + "=" * 60)
    print("生成可视化图表...")
    print("=" * 60)

    # 1. 模型性能对比柱状图
    plot_model_comparison(results)
    print("✓ 模型性能对比图已保存: model_comparison.png")

    # 2. 测试集预测可视化（为每个模型生成）
    for model_name, model in model_instances.items():
        success = visualize_test_predictions(model, X_test, y_test, model_name)
        if success:
            print(f"✓ {model_name}测试集预测可视化已保存: {model_name.replace(' ', '_')}_test_predictions.png")
        else:
            print(f"✗ {model_name}测试集预测可视化失败")

    # 3. 为最佳模型生成混淆矩阵
    best_model_name = max(results, key=results.get)
    best_model = model_instances[best_model_name]
    plot_confusion_matrix_for_best_model(best_model, X_test, y_test, best_model_name)
    print(f"✓ 最佳模型({best_model_name})混淆矩阵已保存: {best_model_name.replace(' ', '_')}_confusion_matrix.png")

    # ==================== 结果汇总 ====================
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)

    for model_name, accuracy in results.items():
        print(f"{model_name:20s}: {accuracy:.4f}")

    best_model = max(results, key=results.get)
    print(f"\n最佳模型: {best_model} (准确率: {results[best_model]:.4f})")

    # 显示生成的文件信息
    print("\n" + "=" * 60)
    print("生成的可视化文件:")
    print("=" * 60)
    print("1. model_comparison.png - 模型性能对比柱状图")
    print("2. knn_hyperparameter.png - KNN超参数选择图")
    for model_name in model_instances.keys():
        print(f"3. {model_name.replace(' ', '_')}_test_predictions.png - {model_name}测试集预测可视化")
    print(f"4. {best_model.replace(' ', '_')}_confusion_matrix.png - 最佳模型混淆矩阵")


if __name__ == "__main__":
    main()