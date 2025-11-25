import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging
from dataclasses import dataclass, field
import pickle
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class Config:
    """优化的配置类"""
    # 数据相关
    data_dir: str = 'ml-100k'
    test_size: float = 0.2
    random_state: int = 42
    implicit_threshold: float = 4.0
    use_cache: bool = True
    cache_dir: str = 'cache'

    # 模型相关
    embedding_dim: int = 128
    hidden_dims: list = field(default_factory=lambda: [256, 128, 64, 32])
    dropout_rates: list = field(default_factory=lambda: [0.4, 0.3, 0.2, 0.1])
    use_batch_norm: bool = True

    # 训练相关
    num_epochs: int = 100
    batch_size: int = 512
    test_batch_size: int = 1024
    learning_rate: float = 0.001
    weight_decay: float = 1e-5

    # 优化策略
    use_class_weights: bool = True
    use_label_smoothing: bool = False  # 【修改】与BCEWithLogitsLoss不兼容
    label_smoothing: float = 0.1

    # 学习率调度
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # DataLoader相关
    num_workers: int = 0
    prefetch_factor: Optional[int] = None  # 【修改】改为可选
    persistent_workers: bool = False

    # 早停相关
    patience: int = 15
    min_delta: float = 0.001

    # 输出相关
    save_dir: str = 'outputs'

    # 混合精度训练
    use_amp: bool = False  # 【新增】

    def __post_init__(self):
        """后初始化处理"""
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # 【修改】自动配置 num_workers
        if self.num_workers == 0:
            import platform
            if platform.system() != 'Windows':
                self.num_workers = min(4, os.cpu_count() or 1)
                self.persistent_workers = self.num_workers > 0
                self.prefetch_factor = 2 if self.num_workers > 0 else None
            else:
                self.prefetch_factor = None


class MovieLensDataset(Dataset):
    """改进的数据集类"""

    def __init__(self, data: pd.DataFrame, num_users: int, num_items: int,
                 config: Config, is_training: bool = True):
        self.num_users = num_users  # 【新增】保存以便数据增强使用
        self.num_items = num_items  # 【新增】
        self.users = torch.tensor(data['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(data['item_id'].values, dtype=torch.long)
        self.ratings = data['rating'].values
        self.is_training = is_training
        self.config = config

        # 计算标签
        labels = (self.ratings >= config.implicit_threshold).astype(np.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

        # 【新增】计算类别权重（供Trainer使用）
        self.class_weights = None
        if is_training and config.use_class_weights:
            self.class_weights = self._compute_class_weights(labels)
            logger.info(f"类别权重: 负类={self.class_weights[0]:.4f}, 正类={self.class_weights[1]:.4f}")

        logger.info(f"{'训练' if is_training else '测试'}集样本数: {len(self)}, "
                    f"正样本比例: {labels.mean():.4f}")

    def _compute_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        """计算类别权重"""
        class_weights = compute_class_weight(
            'balanced',
            classes=np.array([0.0, 1.0]),
            y=labels
        )
        return torch.tensor(class_weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user = self.users[idx]
        item = self.items[idx]
        label = self.labels[idx]

        # 【修改】改进的数据增强，确保不越界
        if self.is_training and torch.rand(1).item() < 0.05:  # 降低概率到5%
            noise = torch.randint(-2, 3, (1,)).item()
            if torch.rand(1).item() < 0.5:
                user = torch.clamp(user + noise, 0, self.num_users - 1).long()
            else:
                item = torch.clamp(item + noise, 0, self.num_items - 1).long()

        return user, item, label


class MovieLensDataManager:
    """数据管理器"""

    def __init__(self, config: Config):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.cache_dir = Path(config.cache_dir)

    def load_movielens_data(self) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]]:
        """加载MovieLens数据集"""

        # 尝试从缓存加载
        if self.config.use_cache:
            cached_data = self._load_from_cache()
            if cached_data is not None:
                logger.info("✅ 从缓存加载数据成功！")
                return cached_data

        logger.info("正在加载MovieLens 100K数据集...")

        # 检查并下载数据
        if not self._check_data_exists():
            if not self._download_data():
                return None

        try:
            # 读取评分数据
            ratings = pd.read_csv(
                self.data_dir / 'u.data',
                sep='\t',
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                dtype={'user_id': np.int32, 'item_id': np.int32, 'rating': np.float32}
            )

            # 读取电影信息
            items = pd.read_csv(
                self.data_dir / 'u.item',
                sep='|',
                encoding='latin-1',
                header=None,
                usecols=[0, 1],
                names=['item_id', 'title']
            )

            # 读取用户信息
            users = pd.read_csv(
                self.data_dir / 'u.user',
                sep='|',
                names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
            )

            num_users = int(ratings['user_id'].max())
            num_items = int(ratings['item_id'].max())

            self._print_statistics(ratings, num_users, num_items)

            # 保存到缓存
            if self.config.use_cache:
                self._save_to_cache(ratings, items, users, num_users, num_items)

            return ratings, items, users, num_users, num_items

        except Exception as e:
            logger.error(f"❌ 加载数据集失败: {e}")
            return None

    def _load_from_cache(self) -> Optional[Tuple]:
        """从缓存加载"""
        cache_file = self.cache_dir / 'movielens_cache.pkl'
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"⚠️ 缓存加载失败: {e}")
        return None

    def _save_to_cache(self, ratings, items, users, num_users, num_items):
        """保存到缓存"""
        cache_file = self.cache_dir / 'movielens_cache.pkl'
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((ratings, items, users, num_users, num_items), f)
            logger.info(f"💾 数据已缓存至 {cache_file}")
        except Exception as e:
            logger.warning(f"⚠️ 缓存保存失败: {e}")

    def _check_data_exists(self) -> bool:
        """检查数据是否存在"""
        required_files = ['u.data', 'u.item', 'u.user']
        return all((self.data_dir / f).exists() for f in required_files)

    def _download_data(self) -> bool:
        """下载数据集"""
        import urllib.request
        import zipfile

        try:
            url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
            logger.info(f"⬇️ 正在从 {url} 下载数据集...")

            zip_path = "ml-100k.zip"
            urllib.request.urlretrieve(url, zip_path)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")

            os.remove(zip_path)
            logger.info("✅ 数据集下载完成！")
            return True

        except Exception as e:
            logger.error(f"❌ 下载失败: {e}")
            return False

    @staticmethod
    def _print_statistics(ratings: pd.DataFrame, num_users: int, num_items: int):
        """打印统计信息"""
        num_ratings = len(ratings)
        sparsity = 1 - num_ratings / (num_users * num_items)

        logger.info(f"📊 数据统计:")
        logger.info(f"  - 用户数量: {num_users:,}")
        logger.info(f"  - 电影数量: {num_items:,}")
        logger.info(f"  - 评分数量: {num_ratings:,}")
        logger.info(f"  - 数据稀疏度: {sparsity:.4f}")

class MatrixFactorizationModel(nn.Module):
    """矩阵分解模型 (MF)"""

    def __init__(self, num_users: int, num_items: int, config: Config, use_sigmoid: bool = True):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users + 1, config.embedding_dim, padding_idx=0)
        self.item_embeddings = nn.Embedding(num_items + 1, config.embedding_dim, padding_idx=0)
        self.use_sigmoid = use_sigmoid
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_embeddings(user)
        item_vec = self.item_embeddings(item)

        if user_vec.dim() > 2:
            user_vec = user_vec.squeeze(1)
            item_vec = item_vec.squeeze(1)

        # 点积
        output = (user_vec * item_vec).sum(dim=-1)

        if self.use_sigmoid:
            output = torch.sigmoid(output)

        return output

class NeuralCollaborativeFiltering(nn.Module):
    """神经协同过滤 (NCF)"""

    def __init__(self, num_users: int, num_items: int, config: Config, use_sigmoid: bool = True):
        super().__init__()
        # GMF部分
        self.gmf_user_embeddings = nn.Embedding(num_users + 1, config.embedding_dim, padding_idx=0)
        self.gmf_item_embeddings = nn.Embedding(num_items + 1, config.embedding_dim, padding_idx=0)

        # MLP部分
        self.mlp_user_embeddings = nn.Embedding(num_users + 1, config.embedding_dim, padding_idx=0)
        self.mlp_item_embeddings = nn.Embedding(num_items + 1, config.embedding_dim, padding_idx=0)

        # MLP层
        mlp_layers = []
        input_dim = config.embedding_dim * 2
        for hidden_dim in [128, 64, 32]:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*mlp_layers)

        # 融合层
        self.output_layer = nn.Linear(config.embedding_dim + 32, 1)
        self.use_sigmoid = use_sigmoid

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        # GMF部分
        gmf_user = self.gmf_user_embeddings(user)
        gmf_item = self.gmf_item_embeddings(item)

        if gmf_user.dim() > 2:
            gmf_user = gmf_user.squeeze(1)
            gmf_item = gmf_item.squeeze(1)

        gmf_output = gmf_user * gmf_item

        # MLP部分
        mlp_user = self.mlp_user_embeddings(user)
        mlp_item = self.mlp_item_embeddings(item)

        if mlp_user.dim() > 2:
            mlp_user = mlp_user.squeeze(1)
            mlp_item = mlp_item.squeeze(1)

        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # 融合
        concat = torch.cat([gmf_output, mlp_output], dim=-1)
        output = self.output_layer(concat).squeeze()

        if self.use_sigmoid:
            output = torch.sigmoid(output)

        return output

class DeepFMModel(nn.Module):
    """DeepFM模型"""

    def __init__(self, num_users: int, num_items: int, config: Config, use_sigmoid: bool = True):
        super().__init__()
        # 共享Embedding
        self.user_embeddings = nn.Embedding(num_users + 1, config.embedding_dim, padding_idx=0)
        self.item_embeddings = nn.Embedding(num_items + 1, config.embedding_dim, padding_idx=0)

        # FM部分 - 一阶
        self.fm_user_bias = nn.Embedding(num_users + 1, 1, padding_idx=0)
        self.fm_item_bias = nn.Embedding(num_items + 1, 1, padding_idx=0)
        self.fm_global_bias = nn.Parameter(torch.zeros(1))

        # Deep部分
        deep_layers = []
        input_dim = config.embedding_dim * 2
        for hidden_dim in [256, 128, 64]:
            deep_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim

        self.deep = nn.Sequential(*deep_layers)
        self.deep_output = nn.Linear(64, 1)

        self.use_sigmoid = use_sigmoid
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        # Embedding
        user_emb = self.user_embeddings(user)
        item_emb = self.item_embeddings(item)

        if user_emb.dim() > 2:
            user_emb = user_emb.squeeze(1)
            item_emb = item_emb.squeeze(1)

        # FM一阶
        fm_first_order = (
                self.fm_user_bias(user).squeeze() +
                self.fm_item_bias(item).squeeze() +
                self.fm_global_bias
        )

        # FM二阶（交互）
        fm_second_order = (user_emb * item_emb).sum(dim=-1)

        # Deep部分
        deep_input = torch.cat([user_emb, item_emb], dim=-1)
        deep_output = self.deep(deep_input)
        deep_output = self.deep_output(deep_output).squeeze()

        # 融合
        output = fm_first_order + fm_second_order + deep_output

        if self.use_sigmoid:
            output = torch.sigmoid(output)

        return output

class ModelComparison:
    """模型对比实验类"""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}

    def get_models(self, num_users: int, num_items: int, use_sigmoid: bool = True):
        """获取所有待对比的模型"""
        models = {
            'MF': MatrixFactorizationModel(num_users, num_items, self.config, use_sigmoid),
            'NCF': NeuralCollaborativeFiltering(num_users, num_items, self.config, use_sigmoid),
            'DeepFM': DeepFMModel(num_users, num_items, self.config, use_sigmoid)        }
        return models

    def train_and_evaluate_model(self, model_name: str, model: nn.Module,
                                 train_loader: DataLoader, test_loader: DataLoader,
                                 train_dataset: MovieLensDataset):
        """训练并评估单个模型"""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"🚀 开始训练模型: {model_name}")
        logger.info(f"{'=' * 60}")

        trainer = Trainer(model, self.config, train_dataset, self.device)
        history = trainer.fit(train_loader, test_loader)

        # 保存结果
        self.results[model_name] = {
            'history': history,
            'best_f1': trainer.best_f1,
            'final_metrics': {
                'accuracy': history['test_accuracies'][-1],
                'f1': history['test_f1_scores'][-1],
                'precision': history['test_precisions'][-1],
                'recall': history['test_recalls'][-1]
            }
        }

        # 保存模型
        trainer.save_model(f'{model_name}_best_model.pth')

        return history

    def run_comparison(self, train_loader: DataLoader, test_loader: DataLoader,
                       train_dataset: MovieLensDataset, num_users: int, num_items: int):
        """运行所有模型的对比实验"""
        use_sigmoid = not (self.config.use_class_weights and train_dataset.class_weights is not None)
        models = self.get_models(num_users, num_items, use_sigmoid)

        for model_name, model in models.items():
            try:
                self.train_and_evaluate_model(
                    model_name, model, train_loader, test_loader, train_dataset
                )
            except Exception as e:
                logger.error(f"❌ 模型 {model_name} 训练失败: {e}")
                continue

        # 生成对比报告
        self.generate_comparison_report()
        self.visualize_comparison()

    def generate_comparison_report(self):
        """生成对比报告"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 模型性能对比报告")
        logger.info("=" * 80)

        # 创建对比表格
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result['final_metrics']
            comparison_data.append({
                '模型': model_name,
                '准确率': f"{metrics['accuracy']:.4f}",
                'F1分数': f"{metrics['f1']:.4f}",
                '精确率': f"{metrics['precision']:.4f}",
                '召回率': f"{metrics['recall']:.4f}",
                '最佳F1': f"{result['best_f1']:.4f}"
            })

        df = pd.DataFrame(comparison_data)
        logger.info(f"\n{df.to_string(index=False)}")

        # 保存到CSV
        df.to_csv(Path(self.config.save_dir) / 'model_comparison.csv', index=False)
        logger.info(f"\n💾 对比结果已保存至 {self.config.save_dir}/model_comparison.csv")

    def visualize_comparison(self):
        """可视化对比结果"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 训练损失对比
        ax1 = fig.add_subplot(gs[0, :])
        for model_name, result in self.results.items():
            epochs = range(1, len(result['history']['train_losses']) + 1)
            ax1.plot(epochs, result['history']['train_losses'],
                     label=model_name, linewidth=2, marker='o', markersize=3)
        ax1.set_title('训练损失对比', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. F1分数对比
        ax2 = fig.add_subplot(gs[1, 0])
        for model_name, result in self.results.items():
            epochs = range(1, len(result['history']['test_f1_scores']) + 1)
            ax2.plot(epochs, result['history']['test_f1_scores'],
                     label=model_name, linewidth=2)
        ax2.set_title('F1分数对比', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 准确率对比
        ax3 = fig.add_subplot(gs[1, 1])
        for model_name, result in self.results.items():
            epochs = range(1, len(result['history']['test_accuracies']) + 1)
            ax3.plot(epochs, result['history']['test_accuracies'],
                     label=model_name, linewidth=2)
        ax3.set_title('准确率对比', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 精确率对比
        ax4 = fig.add_subplot(gs[1, 2])
        for model_name, result in self.results.items():
            epochs = range(1, len(result['history']['test_precisions']) + 1)
            ax4.plot(epochs, result['history']['test_precisions'],
                     label=model_name, linewidth=2)
        ax4.set_title('精确率对比', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Precision')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. 召回率对比
        ax5 = fig.add_subplot(gs[2, 0])
        for model_name, result in self.results.items():
            epochs = range(1, len(result['history']['test_recalls']) + 1)
            ax5.plot(epochs, result['history']['test_recalls'],
                     label=model_name, linewidth=2)
        ax5.set_title('召回率对比', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Recall')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. 最终指标柱状图
        ax6 = fig.add_subplot(gs[2, 1:])
        metrics_names = ['准确率', 'F1分数', '精确率', '召回率']
        x = np.arange(len(metrics_names))
        width = 0.15

        for i, (model_name, result) in enumerate(self.results.items()):
            metrics = result['final_metrics']
            values = [metrics['accuracy'], metrics['f1'],
                      metrics['precision'], metrics['recall']]
            ax6.bar(x + i * width, values, width, label=model_name)

        ax6.set_title('最终性能指标对比', fontsize=12, fontweight='bold')
        ax6.set_ylabel('分数')
        ax6.set_xticks(x + width * 2)
        ax6.set_xticklabels(metrics_names)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')

        plt.savefig(Path(self.config.save_dir) / 'model_comparison.png',
                    dpi=300, bbox_inches='tight')
        logger.info(f"📊 对比可视化图已保存至 {self.config.save_dir}/model_comparison.png")
        plt.close()

class Trainer:
    """训练器"""

    def __init__(self, model: nn.Module, config: Config,
                 train_dataset: MovieLensDataset, device: torch.device):  # 【修复】添加参数
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.train_dataset = train_dataset

        # 【新增】混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

        # 【修改】统一损失函数处理
        if config.use_class_weights and train_dataset.class_weights is not None:
            pos_weight = train_dataset.class_weights[1] / train_dataset.class_weights[0]
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        else:
            self.criterion = nn.BCELoss()

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            verbose=True,
            min_lr=1e-6
        )

        self.early_stopping = EarlyStopping(config.patience, config.min_delta)
        self.best_f1 = 0.0

    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc='Training', leave=False)

        for batch_idx, (user, item, label) in enumerate(pbar):
            user = user.to(self.device)
            item = item.to(self.device)
            label = label.to(self.device)

            self.optimizer.zero_grad()

            # 【新增】混合精度训练支持
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    predictions = self.model(user, item)
                    loss = self.criterion(predictions, label)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(user, item)
                loss = self.criterion(predictions, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        for user, item, label in tqdm(test_loader, desc='Evaluating', leave=False):
            user = user.to(self.device)
            item = item.to(self.device)

            predictions = self.model(user, item)

            # 【修改】统一处理概率输出
            if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                probabilities = torch.sigmoid(predictions)
            else:
                probabilities = predictions

            probabilities = probabilities.cpu().numpy()
            binary_preds = (probabilities > 0.5).astype(np.float32)

            all_predictions.extend(binary_preds)
            all_probabilities.extend(probabilities)
            all_labels.extend(label.numpy())

        # 寻找最优阈值
        optimal_threshold = self._find_optimal_threshold(all_probabilities, all_labels)
        optimized_preds = (np.array(all_probabilities) > optimal_threshold).astype(np.float32)

        return {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'optimized_accuracy': accuracy_score(all_labels, optimized_preds),
            'f1': f1_score(all_labels, all_predictions, zero_division=0),
            'optimized_f1': f1_score(all_labels, optimized_preds, zero_division=0),
            'precision': precision_score(all_labels, all_predictions, zero_division=0),
            'recall': recall_score(all_labels, all_predictions, zero_division=0),
            'optimal_threshold': optimal_threshold
        }

    @staticmethod
    def _find_optimal_threshold(probabilities, labels, n_thresholds=100):
        """寻找最优阈值"""
        best_f1 = 0
        best_threshold = 0.5

        for threshold in np.linspace(0.3, 0.7, n_thresholds):
            preds = (np.array(probabilities) > threshold).astype(np.float32)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold

    def fit(self, train_loader: DataLoader, test_loader: DataLoader) -> Dict:
        """训练模型"""
        history = {
            'train_losses': [], 'test_accuracies': [], 'test_f1_scores': [],
            'test_precisions': [], 'test_recalls': [], 'learning_rates': [],
            'optimal_thresholds': []
        }

        for epoch in range(self.config.num_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 评估
            metrics = self.evaluate(test_loader)

            # 记录历史
            history['train_losses'].append(train_loss)
            history['test_accuracies'].append(metrics['accuracy'])
            history['test_f1_scores'].append(metrics['f1'])
            history['test_precisions'].append(metrics['precision'])
            history['test_recalls'].append(metrics['recall'])
            history['learning_rates'].append(current_lr)
            history['optimal_thresholds'].append(metrics['optimal_threshold'])

            # 学习率调度（使用标准F1，更稳定）
            self.scheduler.step(metrics['f1'])

            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Loss: {train_loss:.4f} | Acc: {metrics['accuracy']:.4f} | "
                f"F1: {metrics['f1']:.4f} (Opt: {metrics['optimized_f1']:.4f}) | "
                f"P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f} | "
                f"Thr: {metrics['optimal_threshold']:.3f} | LR: {current_lr:.6f}"
            )

            # 保存最佳模型
            if metrics['f1'] > self.best_f1:
                self.best_f1 = metrics['f1']
                self.save_model('best_model.pth')
                logger.info("🔥 新的最佳模型！")

            # 【修复】使用EarlyStopping类
            if self.early_stopping(metrics['f1']):
                logger.info(f"🛑 早停触发于epoch {epoch + 1}")
                break

            # 学习率过低停止
            if current_lr < 1e-6:
                logger.info("🛑 学习率过低，停止训练")
                break

        return history

    def save_model(self, filename: str):
        """保存模型"""
        save_path = Path(self.config.save_dir) / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_f1': self.best_f1
        }, save_path)

class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 5, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            return False

        if val_score < self.best_score + self.min_delta:
            self.counter += 1
            return self.counter >= self.patience
        else:
            self.best_score = val_score
            self.counter = 0
            return False

class Visualizer:
    """可视化"""

    @staticmethod
    def plot_training_results(history: Dict, save_path: str = 'training_results.png'):
        """绘制训练结果"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        epochs = range(1, len(history['train_losses']) + 1)

        # 训练损失
        axes[0, 0].plot(epochs, history['train_losses'], 'b-', linewidth=2)
        axes[0, 0].set_title('训练损失', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # 准确率
        axes[0, 1].plot(epochs, history['test_accuracies'], 'g-', linewidth=2)
        axes[0, 1].set_title('测试准确率', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)

        # F1分数
        axes[0, 2].plot(epochs, history['test_f1_scores'], 'r-', linewidth=2)
        axes[0, 2].set_title('F1分数', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].grid(True, alpha=0.3)

        # 精确率和召回率
        axes[1, 0].plot(epochs, history['test_precisions'], 'c-', label='精确率', linewidth=2)
        axes[1, 0].plot(epochs, history['test_recalls'], 'm-', label='召回率', linewidth=2)
        axes[1, 0].set_title('精确率与召回率', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 学习率
        axes[1, 1].plot(epochs, history['learning_rates'], 'orange', linewidth=2)
        axes[1, 1].set_title('学习率变化', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)

        # 最优阈值
        axes[1, 2].plot(epochs, history['optimal_thresholds'], 'purple', linewidth=2)
        axes[1, 2].set_title('最优分类阈值', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Threshold')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 训练结果图已保存至 {save_path}")
        plt.close()


def main():
    """主函数 - 模型对比实验版本"""
    logger.info("=" * 80)
    logger.info("🚀 深度学习推荐系统模型对比实验 - MovieLens 100K")
    logger.info("=" * 80)

    # 1. 初始化配置
    config = Config()
    config.num_epochs = 50  # 减少epoch以加快对比实验

    # 2. 加载数据
    data_manager = MovieLensDataManager(config)
    result = data_manager.load_movielens_data()

    if result is None:
        logger.error("❌ 数据加载失败，程序终止")
        return

    ratings, items, users, num_users, num_items = result

    # 3. 划分数据集
    logger.info("📊 正在划分数据集...")
    train_data, test_data = train_test_split(
        ratings,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=(ratings['rating'] >= config.implicit_threshold).astype(int)
    )

    # 4. 创建数据集
    train_dataset = MovieLensDataset(train_data, num_users, num_items, config, is_training=True)
    test_dataset = MovieLensDataset(test_data, num_users, num_items, config, is_training=False)

    # 5. 创建数据加载器
    dataloader_kwargs = {
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'pin_memory': torch.cuda.is_available(),
    }

    if config.num_workers > 0 and config.prefetch_factor is not None:
        dataloader_kwargs['prefetch_factor'] = config.prefetch_factor
        dataloader_kwargs['persistent_workers'] = config.persistent_workers

    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # 6. 运行模型对比实验
    comparison = ModelComparison(config)
    comparison.run_comparison(train_loader, test_loader, train_dataset, num_users, num_items)

    logger.info("\n" + "=" * 80)
    logger.info("✅ 所有模型对比实验完成！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

