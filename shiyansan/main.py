import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
import pandas as pd
import sys
import warnings

warnings.filterwarnings('ignore')

# ============ 性能优化配置 ============
if torch.cuda.is_available():
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 设置中文字体（增强兼容性）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 定义优化后的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)  # 添加dropout防止过拟合

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # 在全连接层后添加dropout
        x = self.fc2(x)
        return x


def get_data_loaders(batch_size_train=256, batch_size_test=512, num_workers=4):
    """数据加载器工厂函数（增强版）"""
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 加载数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)

    # 根据平台调整参数
    is_cuda = torch.cuda.is_available()

    # 创建数据加载器
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers if sys.platform != 'win32' else 0,
        pin_memory=is_cuda,
        persistent_workers=True if (num_workers > 0 and sys.platform != 'win32') else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=max(2, num_workers // 2) if sys.platform != 'win32' else 0,
        pin_memory=is_cuda,
        persistent_workers=True if (num_workers > 0 and sys.platform != 'win32') else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    return trainloader, testloader, trainset, testset


def train_model(model, trainloader, testloader, device, num_epochs=10):
    """训练函数（完全修复版）"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加标签平滑
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=num_epochs,
        steps_per_epoch=len(trainloader)
    )
    scaler = GradScaler() if device.type == 'cuda' else None

    # 记录训练过程
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    learning_rates = []
    epoch_times = []

    print(f"\n{'=' * 60}")
    print(f"开始训练 - 总轮次: {num_epochs}, 设备: {device}")
    print(f"{'=' * 60}\n")

    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        # 创建进度条
        pbar = tqdm(trainloader, desc=f'Epoch [{epoch + 1}/{num_epochs}]',
                    ncols=100, file=sys.stdout, leave=True)

        for images, labels in pbar:
            if device.type == 'cuda':
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
            else:
                images = images.to(device)
                labels = labels.to(device)

            # 混合精度训练
            if scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            scheduler.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            current_lr = scheduler.get_last_lr()[0]
            learning_rates.append(current_lr)
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100. * correct / total:.2f}%',
                'lr': f'{current_lr:.1e}'
            })

        pbar.close()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # 计算epoch统计
        avg_loss = running_loss / len(trainloader)
        train_accuracy = 100. * correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)

        # 测试集评估
        test_accuracy = evaluate_model(model, testloader, device, verbose=False)
        test_accuracies.append(test_accuracy)

        # 打印epoch摘要
        print(f"Epoch [{epoch + 1}/{num_epochs}] 完成 | "
              f"时间: {epoch_time:.1f}s | "
              f"训练Loss: {avg_loss:.4f} | "
              f"训练Acc: {train_accuracy:.2f}% | "
              f"测试Acc: {test_accuracy:.2f}%\n")

    print(f"{'=' * 60}")
    print(f"训练完成！")
    print(f"{'=' * 60}\n")

    return model, {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'learning_rates': learning_rates,
        'epoch_times': epoch_times
    }


def evaluate_model(model, testloader, device, use_amp=True, verbose=True, return_details=False):
    """评估函数（优化版）"""
    use_amp = use_amp and (device.type == 'cuda')
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        iterator = tqdm(testloader, desc='评估中', ncols=80, disable=not verbose)

        for images, labels in iterator:
            if device.type == 'cuda':
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
            else:
                images = images.to(device)
                labels = labels.to(device)

            if use_amp:
                with autocast():
                    outputs = model(images)
            else:
                outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if return_details:
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total

    if verbose:
        print(f"测试准确率: {accuracy:.2f}%")

    if return_details:
        return accuracy, all_predictions, all_labels
    else:
        return accuracy


def plot_training_history(history, model_name="CNN模型"):
    """绘制训练历史图表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(history['train_losses']) + 1)

    # 1. 训练损失
    ax1.plot(epochs, history['train_losses'], 'b-', label='训练损失', linewidth=2)
    ax1.set_xlabel('训练轮次', fontsize=12)
    ax1.set_ylabel('损失值', fontsize=12)
    ax1.set_title(f'{model_name} - 训练损失曲线', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. 训练和测试准确率
    ax2.plot(epochs, history['train_accuracies'], 'g-', label='训练准确率', linewidth=2)
    ax2.plot(epochs, history['test_accuracies'], 'r-', label='测试准确率', linewidth=2)
    ax2.set_xlabel('训练轮次', fontsize=12)
    ax2.set_ylabel('准确率 (%)', fontsize=12)
    ax2.set_title(f'{model_name} - 准确率曲线', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. 学习率变化
    steps_per_epoch = len(history['learning_rates']) // len(epochs)
    lr_per_epoch = [history['learning_rates'][min((i + 1) * steps_per_epoch - 1,
                                                  len(history['learning_rates']) - 1)] for i in range(len(epochs))]
    ax3.plot(epochs, lr_per_epoch, 'purple', label='学习率', linewidth=2)
    ax3.set_xlabel('训练轮次', fontsize=12)
    ax3.set_ylabel('学习率', fontsize=12)
    ax3.set_title(f'{model_name} - 学习率变化', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. 每轮训练时间
    ax4.bar(epochs, history['epoch_times'], color='orange', alpha=0.7)
    ax4.set_xlabel('训练轮次', fontsize=12)
    ax4.set_ylabel('时间 (秒)', fontsize=12)
    ax4.set_title(f'{model_name} - 每轮训练时间', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("✓ 训练历史图表已保存: training_history.png")
    plt.show()


def plot_confusion_matrix(all_predictions, all_labels, class_names, model_name="CNN模型"):
    """绘制混淆矩阵"""
    cm = confusion_matrix(all_labels, all_predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': '数量'})
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title(f'{model_name} - 混淆矩阵', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ 混淆矩阵已保存: confusion_matrix.png")
    plt.show()

    # 打印分类报告
    print("\n" + "=" * 60)
    print("分类报告:")
    print("=" * 60)
    print(classification_report(all_labels, all_predictions,
                                target_names=class_names, digits=4))


def plot_sample_predictions(model, testset, device, class_names, num_samples=10):
    """绘制样本预测结果"""
    model.eval()

    # 随机选择样本
    indices = np.random.choice(len(testset), num_samples, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('样本预测结果可视化', fontsize=16, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            image, true_label = testset[indices[i]]

            # 预测
            with torch.no_grad():
                image_tensor = image.unsqueeze(0).to(device)
                if device.type == 'cuda':
                    image_tensor = image_tensor.to(memory_format=torch.channels_last)
                    with autocast():
                        output = model(image_tensor)
                else:
                    output = model(image_tensor)
                _, predicted = torch.max(output, 1)
                predicted_label = predicted.item()

            # 反归一化显示图像
            image = image.numpy().transpose(1, 2, 0)
            image = np.clip(image * np.array([0.2023, 0.1994, 0.2010]) +
                            np.array([0.4914, 0.4822, 0.4465]), 0, 1)

            ax.imshow(image)
            color = 'green' if true_label == predicted_label else 'red'
            ax.set_title(f'真实: {class_names[true_label]}\n预测: {class_names[predicted_label]}',
                         color=color, fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
    print("✓ 样本预测图已保存: sample_predictions.png")
    plt.show()


def generate_performance_report(history, final_accuracy, total_training_time, model_name="CNN模型"):
    """生成性能报告"""
    report = f"""
{'=' * 70}
{model_name} - 训练性能报告
{'=' * 70}

训练配置:
  • 训练轮次: {len(history['train_losses'])}
  • 最终训练准确率: {history['train_accuracies'][-1]:.2f}%
  • 最终测试准确率: {final_accuracy:.2f}%
  • 总训练时间: {total_training_time:.2f} 秒 ({total_training_time / 60:.2f} 分钟)
  • 平均每轮时间: {np.mean(history['epoch_times']):.2f} 秒

训练过程统计:
  • 最小训练损失: {min(history['train_losses']):.4f}
  • 最大训练准确率: {max(history['train_accuracies']):.2f}%
  • 最大测试准确率: {max(history['test_accuracies']):.2f}%
  • 学习率范围: {min(history['learning_rates']):.6f} - {max(history['learning_rates']):.6f}
  • 过拟合程度: {max(history['train_accuracies']) - max(history['test_accuracies']):.2f}%

生成的文件:
  ✓ training_history.png - 训练历史曲线
  ✓ confusion_matrix.png - 混淆矩阵
  ✓ sample_predictions.png - 样本预测可视化
  ✓ model_trained.pth - 训练好的模型
  ✓ training_report.txt - 本报告文本版本

{'=' * 70}
"""

    print(report)

    # 保存报告到文件
    with open('training_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("✓ 训练报告已保存: training_report.txt\n")


# ============ 主程序入口 ============
if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    print(f"{'=' * 60}\n")

    # 获取数据加载器
    print("正在加载数据集...")
    trainloader, testloader, trainset, testset = get_data_loaders()
    print(f"✓ 训练集大小: {len(trainset)}")
    print(f"✓ 测试集大小: {len(testset)}")

    # 类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 初始化模型
    print("\n正在初始化模型...")
    model = SimpleCNN(num_classes=10).to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    print(f"✓ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型（建议使用10-20轮，不要50轮）
    start_time = time.time()
    model, history = train_model(model, trainloader, testloader, device, num_epochs=10)
    total_training_time = time.time() - start_time

    # 最终评估
    print("正在进行最终评估...")
    final_accuracy, all_predictions, all_labels = evaluate_model(
        model, testloader, device, return_details=True, verbose=True
    )

    # 保存模型
    torch.save(model.state_dict(), 'model_trained.pth')
    print("\n✓ 模型已保存: model_trained.pth")

    # 生成可视化图表
    print("\n正在生成可视化图表...")
    plot_training_history(history, "CIFAR-10 CNN")
    plot_confusion_matrix(all_predictions, all_labels, class_names, "CIFAR-10 CNN")
    plot_sample_predictions(model, testset, device, class_names)

    # 生成性能报告
    generate_performance_report(history, final_accuracy, total_training_time, "CIFAR-10 CNN")


    # 单张图片推理示例
    @torch.no_grad()
    def predict_single_image(model, image_tensor):
        model.eval()
        image_tensor = image_tensor.unsqueeze(0).to(device)
        if device.type == 'cuda':
            image_tensor = image_tensor.to(memory_format=torch.channels_last)
            with autocast():
                output = model(image_tensor)
        else:
            output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]


    # 测试单张图片
    test_image, test_label = testset[0]
    prediction = predict_single_image(model, test_image)
    print(f"\n单张图片测试:")
    print(f"  预测类别: {prediction}")
    print(f"  真实类别: {class_names[test_label]}")
    print(f"  结果: {'✓ 正确' if prediction == class_names[test_label] else '✗ 错误'}")

    print(f"\n{'=' * 60}")
    print("所有任务已完成！")
    print(f"{'=' * 60}\n")
