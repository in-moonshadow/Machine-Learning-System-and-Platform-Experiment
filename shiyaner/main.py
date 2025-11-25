import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 1. 定义更深的卷积神经网络（充分利用GPU并行计算）
class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        # 更深的卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 2. 准备数据集（优化数据加载）
def prepare_data(batch_size=512, num_workers=4):  # 增大batch_size，增加workers
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # 使用pin_memory加速GPU数据传输
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


# 3. 优化的训练函数（使用混合精度训练）
def train(model, device, train_loader, optimizer, criterion, epoch, use_amp=False):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None

    start_time = time.time()

    # 使用进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', ncols=100)

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler:
            # 混合精度训练
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 普通训练
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })

    end_time = time.time()
    epoch_time = end_time - start_time

    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total

    return epoch_time, avg_loss, accuracy


# 4. 优化的测试函数
def test(model, device, test_loader, criterion, use_amp=False):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    start_time = time.time()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
            else:
                output = model(data)
                loss = criterion(output, target)

            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    end_time = time.time()
    test_time = end_time - start_time

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total

    print(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')

    return test_time, avg_loss, accuracy


# 5. GPU优化配置函数
def setup_gpu_optimizations():
    """设置GPU优化参数"""
    if torch.cuda.is_available():
        # 启用TF32数学模式（在Ampere架构及以上的GPU上）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # 启用cudnn自动调优
        torch.backends.cudnn.benchmark = True

        # 设置GPU设备属性
        torch.cuda.set_per_process_memory_fraction(0.8)  # 限制内存使用，避免OOM

        print("GPU优化已启用:")
        print(f"- TF32数学模式: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"- cuDNN自动调优: {torch.backends.cudnn.benchmark}")


# 6. 可视化结果函数
def plot_results(results, filename="gpu_cpu_comparison.png"):
    """绘制性能对比图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    devices = ['cpu', 'cuda'] if 'cuda' in results else ['cpu']
    colors = {'cpu': 'blue', 'cuda': 'red'}

    # 1. 总时间对比
    total_times = [results[device]['total_time'] for device in devices]
    ax1.bar(devices, total_times, color=[colors[d] for d in devices], alpha=0.7)
    ax1.set_title('总训练时间对比')
    ax1.set_ylabel('时间 (秒)')
    for i, v in enumerate(total_times):
        ax1.text(i, v + max(total_times) * 0.01, f'{v:.1f}s', ha='center')

    # 2. 每轮训练时间对比
    avg_train_times = [results[device]['avg_train_time'] for device in devices]
    ax2.bar(devices, avg_train_times, color=[colors[d] for d in devices], alpha=0.7)
    ax2.set_title('平均每轮训练时间')
    ax2.set_ylabel('时间 (秒)')
    for i, v in enumerate(avg_train_times):
        ax2.text(i, v + max(avg_train_times) * 0.01, f'{v:.1f}s', ha='center')

    # 3. 加速比
    if 'cuda' in results:
        speedups = {
            '总时间': results['cpu']['total_time'] / results['cuda']['total_time'],
            '训练时间': results['cpu']['avg_train_time'] / results['cuda']['avg_train_time'],
            '测试时间': results['cpu']['avg_test_time'] / results['cuda']['avg_test_time']
        }
        ax3.bar(speedups.keys(), speedups.values(), color='green', alpha=0.7)
        ax3.set_title('GPU加速比')
        ax3.set_ylabel('加速倍数')
        for i, v in enumerate(speedups.values()):
            ax3.text(i, v + 0.1, f'{v:.1f}x', ha='center')

    # 4. 准确率对比
    train_acc = [results[device]['final_train_acc'] for device in devices]
    test_acc = [results[device]['final_test_acc'] for device in devices]
    x = np.arange(len(devices))
    width = 0.35
    ax4.bar(x - width / 2, train_acc, width, label='训练准确率', alpha=0.7)
    ax4.bar(x + width / 2, test_acc, width, label='测试准确率', alpha=0.7)
    ax4.set_title('准确率对比')
    ax4.set_ylabel('准确率 (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(devices)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# 7. 主函数：对比CPU和GPU性能（优化版本）
def compare_cpu_gpu_optimized(num_epochs=5, use_amp=True, large_batch=True):
    """优化版的CPU/GPU对比函数"""

    # 根据是否使用大batch设置参数
    batch_size = 1024 if large_batch else 64
    num_workers = 8 if large_batch else 4

    print("准备数据...")
    train_loader, test_loader = prepare_data(batch_size=batch_size, num_workers=num_workers)

    # 检查GPU是否可用
    use_cuda = torch.cuda.is_available()
    print(f"CUDA Available: {use_cuda}")
    if use_cuda:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        setup_gpu_optimizations()

    devices = ['cpu']
    if use_cuda:
        devices.append('cuda')

    results = {}

    for device_name in devices:
        print(f"\n{'=' * 60}")
        print(f"在 {device_name.upper()} 上训练（批次大小: {batch_size}）")
        if device_name == 'cuda' and use_amp:
            print("使用混合精度训练 (AMP)")
        print(f"{'=' * 60}")

        device = torch.device(device_name)

        # 创建更深的模型
        model = DeepCNN().to(device)

        # 根据设备选择优化器参数
        if device_name == 'cuda':
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.001)

        criterion = nn.CrossEntropyLoss()

        # 记录时间
        train_times = []
        test_times = []
        train_accuracies = []
        test_accuracies = []

        total_start = time.time()

        # 训练和测试
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            train_time, train_loss, train_acc = train(
                model, device, train_loader, optimizer, criterion, epoch,
                use_amp=use_amp and device_name == 'cuda'
            )

            test_time, test_loss, test_acc = test(
                model, device, test_loader, criterion,
                use_amp=use_amp and device_name == 'cuda'
            )

            epoch_time = time.time() - epoch_start

            train_times.append(train_time)
            test_times.append(test_time)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            print(f"Epoch {epoch} - 训练时间: {train_time:.2f}s, "
                  f"测试时间: {test_time:.2f}s, 总时间: {epoch_time:.2f}s")
            print(f"训练准确率: {train_acc:.2f}%, 测试准确率: {test_acc:.2f}%\n")

        total_time = time.time() - total_start

        # 保存结果
        results[device_name] = {
            'total_time': total_time,
            'avg_train_time': np.mean(train_times),
            'avg_test_time': np.mean(test_times),
            'final_train_acc': train_accuracies[-1],
            'final_test_acc': test_accuracies[-1],
            'train_times': train_times,
            'test_times': test_times,
            'batch_size': batch_size
        }

        # 清理GPU内存
        if device_name == 'cuda':
            torch.cuda.empty_cache()

    # 8. 打印详细对比结果
    print(f"\n{'=' * 60}")
    print("性能对比总结")
    print(f"{'=' * 60}")

    for device_name in devices:
        r = results[device_name]
        print(f"\n{device_name.upper()} 结果:")
        print(f"  总时间: {r['total_time']:.2f}秒")
        print(f"  平均每轮训练时间: {r['avg_train_time']:.2f}秒")
        print(f"  平均每轮测试时间: {r['avg_test_time']:.2f}秒")
        print(f"  最终训练准确率: {r['final_train_acc']:.2f}%")
        print(f"  最终测试准确率: {r['final_test_acc']:.2f}%")
        print(f"  批次大小: {r['batch_size']}")

    # 计算加速比
    if 'cuda' in results and 'cpu' in results:
        speedup_train = results['cpu']['avg_train_time'] / results['cuda']['avg_train_time']
        speedup_test = results['cpu']['avg_test_time'] / results['cuda']['avg_test_time']
        speedup_total = results['cpu']['total_time'] / results['cuda']['total_time']

        print(f"\nGPU 加速效果:")
        print(f"  训练: {speedup_train:.2f}x 更快")
        print(f"  测试: {speedup_test:.2f}x 更快")
        print(f"  总体: {speedup_total:.2f}x 更快")

        # 生成可视化图表
        plot_results(results)

    return results


# 8. 对比实验
def run_comprehensive_comparison():
    """运行全面的对比实验"""

    print("开始全面的CPU/GPU性能对比实验...")

    # # 实验1: 基础配置（小批次）
    # print("\n" + "=" * 70)
    # print("实验1: 基础配置（批次大小=64）")
    # print("=" * 70)
    # results_basic = compare_cpu_gpu_optimized(num_epochs=3, use_amp=False, large_batch=False)

    # 实验2: 优化配置（大批次 + 混合精度）
    if torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("实验1: 优化配置（批次大小=1024 + 混合精度）")
        print("=" * 70)
        results_optimized = compare_cpu_gpu_optimized(num_epochs=5, use_amp=True, large_batch=True)

        return results_basic, results_optimized

    return results_basic, None


# 运行对比实验
if __name__ == '__main__':
    results_basic, results_optimized = run_comprehensive_comparison()

    # 如果有优化结果，打印最终对比
    if results_optimized:
        print("\n" + "=" * 70)
        print("最终优化效果对比")
        print("=" * 70)

        basic_speedup = (results_basic['cpu']['avg_train_time'] /
                         results_basic['cuda']['avg_train_time'] if 'cuda' in results_basic else 1)
        optimized_speedup = (results_optimized['cpu']['avg_train_time'] /
                             results_optimized['cuda']['avg_train_time'])

        print(f"基础配置加速比: {basic_speedup:.2f}x")
        print(f"优化配置加速比: {optimized_speedup:.2f}x")
        print(f"优化效果提升: {optimized_speedup / basic_speedup:.2f}x")