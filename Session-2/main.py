"""
MNIST Classification - CNN with MPS Acceleration
Author: NoGamble
Device: Mac with M2 chip (MPS)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")


train_data = datasets.MNIST(root='./Session-1/data', train=True, download=False, transform=transform)
test_data  = datasets.MNIST(root='./Session-1/data', train=False, transform=transform)

# 划分训练集(80%)和验证集(20%)
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = random_split(train_data, [train_size, val_size])

# 创建DataLoader
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1000, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

# 初始化模型并转移到设备
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# 训练函数
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = 100. * correct / len(train_data)
    return train_loss, train_acc

# 验证函数
def validate():
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / len(val_data)
    return val_loss, val_acc

# 测试函数
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / len(test_data)
    return test_loss, test_acc

# 训练监控
train_losses, train_accs = [], []
val_losses, val_accs = [], []
best_val_acc = 0
patience = 3
trigger_times = 0

# 打印初始信息
print(f"Training Samples: {len(train_data)} | Validation Samples: {len(val_data)} | Test Samples: {len(test_data)}")
print("=" * 60)

# 训练循环
for epoch in range(1, 31):
    # 训练阶段
    train_loss, train_acc = train(epoch)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # 验证阶段
    val_loss, val_acc = validate()
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # 打印进度
    print(f'Epoch {epoch:02d}: '
          f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
          f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
    
    # 早停机制
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        trigger_times = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f'Early stopping triggered at epoch {epoch}')
            break

# 加载最佳模型进行最终测试
model.load_state_dict(torch.load('best_model.pth'))
final_test_loss, final_test_acc = test()
print("=" * 60)
print(f'【Final Test Results】 Loss: {final_test_loss:.4f} | Accuracy: {final_test_acc:.2f}%')

# 可视化训练过程
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train')
plt.plot(val_accs, label='Validation')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()

# 可视化测试集错误样本
def show_errors():
    model.eval()
    wrong_samples = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            mask = pred != target
            wrong_samples.extend(zip(data[mask].cpu(), pred[mask].cpu(), target[mask].cpu()))
    
    # 显示前6个错误样本
    plt.figure(figsize=(12, 3))
    for i in range(min(6, len(wrong_samples))):
        img, pred, true = wrong_samples[i]
        plt.subplot(1, 6, i+1)
        plt.imshow(img[0], cmap='gray')
        plt.title(f'P: {pred.item()}\nT: {true.item()}', color='red' if pred != true else 'green')
        plt.axis('off')
    plt.suptitle('Test Set Error Samples', y=1.1)
    plt.savefig('error_samples.png')
    plt.show()

show_errors()