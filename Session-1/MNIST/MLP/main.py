"""
MNIST Handwritten Digit Classification - Logistic Regression with Multiple Optimizers
Author: Yujing Xiong
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import platform

# 设置全局样式
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'Arial'

# 1. 数据准备（兼容MPS设备）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def load_data():
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    
    # 划分训练/验证集 (6:2:2)
    train_size = int(0.6 * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])
    
    return train_data, val_data, test_data

# 2. 定义MLP模型（针对M2优化）
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

# 3. 训练和评估函数（支持MPS设备）
def train_model(optimizer_name, device='mps', lr=0.001, epochs=10):
    # 初始化
    train_data, val_data, test_data = load_data()
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=128)
    test_loader = DataLoader(test_data, batch_size=128)
    
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 优化器选择
    optimizers = {
        'SGD': optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        'Adam': optim.Adam(model.parameters(), lr=lr),
        'RMSprop': optim.RMSprop(model.parameters(), lr=lr, alpha=0.99),
        'Adagrad': optim.Adagrad(model.parameters(), lr=lr)
    }
    optimizer = optimizers[optimizer_name]
    
    # 记录数据
    train_losses, val_accuracies, epoch_times = [], [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f'{optimizer_name} Epoch {epoch+1}/{epochs}')
        
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # 验证集评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 记录指标
        train_losses.append(running_loss/len(train_loader))
        val_accuracies.append(correct/total)
        epoch_times.append(progress.format_dict['elapsed'])
    
    # 测试集评估
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    return {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'test_accuracy': test_correct/test_total,
        'training_time': sum(epoch_times)
    }

# 4. 主执行函数
def main():
    # 检查设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"System: {platform.system()} {platform.machine()}")
    
    # 比较优化器
    optimizers = ['SGD', 'Adam', 'RMSprop', 'Adagrad']
    results = {}
    
    for opt in optimizers:
        print(f"\n=== Training with {opt} ===")
        results[opt] = train_model(opt, device=device)
    
    # 可视化结果
    visualize_results(results)

# 5. 专业可视化（Seaborn）
def visualize_results(results):
    # 准备数据
    metrics = []
    for opt in results:
        for epoch, loss in enumerate(results[opt]['train_losses']):
            metrics.append({
                'Optimizer': opt,
                'Epoch': epoch+1,
                'Metric': 'Training Loss',
                'Value': loss
            })
        for epoch, acc in enumerate(results[opt]['val_accuracies']):
            metrics.append({
                'Optimizer': opt,
                'Epoch': epoch+1,
                'Metric': 'Validation Accuracy',
                'Value': acc
            })
    
    df = pd.DataFrame(metrics)
    
    # 创建画布
    plt.figure(figsize=(16, 12))
    plt.suptitle('MNIST MLP Performance on M2 MacBook Air', y=1.02, fontsize=16)
    
    # 1. 训练损失曲线
    plt.subplot(2, 2, 1)
    sns.lineplot(
        data=df[df['Metric'] == 'Training Loss'],
        x='Epoch', y='Value', hue='Optimizer',
        palette="viridis", linewidth=2.5
    )
    plt.title('Training Loss', pad=12)
    plt.xlabel('Epoch', labelpad=10)
    plt.ylabel('Loss', labelpad=10)
    
    # 2. 验证准确率曲线
    plt.subplot(2, 2, 2)
    sns.lineplot(
        data=df[df['Metric'] == 'Validation Accuracy'],
        x='Epoch', y='Value', hue='Optimizer',
        palette="plasma", linewidth=2.5
    )
    plt.title('Validation Accuracy', pad=12)
    plt.xlabel('Epoch', labelpad=10)
    plt.ylabel('Accuracy', labelpad=10)
    plt.ylim(0.85, 1.0)
    
    # 3. 测试准确率对比
    plt.subplot(2, 2, 3)
    test_acc = {opt: results[opt]['test_accuracy'] for opt in results}
    ax = sns.barplot(
        x=list(test_acc.keys()), 
        y=list(test_acc.values()),
        palette="mako"
    )
    plt.title('Final Test Accuracy', pad=12)
    plt.xlabel('Optimizer', labelpad=10)
    plt.ylabel('Accuracy', labelpad=10)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.4f}", 
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center', xytext=(0, 10), textcoords='offset points'
        )
    
    # 4. 训练时间对比
    plt.subplot(2, 2, 4)
    times = {opt: results[opt]['training_time'] for opt in results}
    ax = sns.barplot(
        x=list(times.keys()), 
        y=list(times.values()),
        palette="rocket"
    )
    plt.title('Total Training Time', pad=12)
    plt.xlabel('Optimizer', labelpad=10)
    plt.ylabel('Time (seconds)', labelpad=10)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}s", 
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center', xytext=(0, 10), textcoords='offset points'
        )
    
    plt.tight_layout()
    plt.savefig('mnist_mlp_m2_results.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()