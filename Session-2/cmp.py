import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 设备设置
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载数据
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

# L1正则化器
class L1Regularizer:
    def __init__(self, lambda_l1=1e-5):
        self.lambda_l1 = lambda_l1
    
    def __call__(self, model):
        l1_loss = 0
        for param in model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.lambda_l1 * l1_loss

# 训练函数（通用）
def train(model, optimizer, criterion, l1_reg=None):
    model.train()
    train_loss = 0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 添加L1正则项（如果启用）
        if l1_reg is not None:
            loss += l1_reg(model)
            
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = 100. * correct / len(train_data)
    return train_loss, train_acc

# 验证函数
def validate(model, criterion):
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
def test(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    test_acc = 100. * correct / len(test_data)
    return test_acc

# 比较正则化方法
def compare_regularization():
    results = {}
    histories = {}
    
    for reg_type in ['none', 'l1', 'l2']:
        print(f"\n=== Training with {reg_type.upper()} regularization ===")
        model = CNN().to(device)
        criterion = nn.CrossEntropyLoss()
        
        if reg_type == 'l1':
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            l1_reg = L1Regularizer(lambda_l1=1e-5)
        elif reg_type == 'l2':
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            l1_reg = None
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            l1_reg = None
        
        # 训练监控
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        best_val_acc = 0
        
        # 训练循环
        for epoch in range(1, 16):
            train_loss, train_acc = train(model, optimizer, criterion, l1_reg)
            val_loss, val_acc = validate(model, criterion)
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_model_{reg_type}.pth')
            
            print(f'Epoch {epoch:02d}: '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # 保存训练历史
        histories[reg_type] = {
            'train_loss': train_losses,
            'train_acc': train_accs,
            'val_loss': val_losses,
            'val_acc': val_accs
        }
        
        # 测试最佳模型
        model.load_state_dict(torch.load(f'best_model_{reg_type}.pth'))
        test_acc = test(model)
        results[reg_type] = test_acc
        print(f'{reg_type.upper()} Test Accuracy: {test_acc:.2f}%')
    
    return results, histories

# 运行比较
results, histories = compare_regularization()

# 可视化训练曲线
plt.figure(figsize=(15, 5))
colors = {'none': 'blue', 'l1': 'red', 'l2': 'green'}

# 绘制损失曲线
plt.subplot(1, 2, 1)
for reg_type in histories:
    plt.plot(histories[reg_type]['train_loss'], 
             label=f'{reg_type.upper()} Train', 
             color=colors[reg_type], 
             linestyle='-')
    plt.plot(histories[reg_type]['val_loss'], 
             label=f'{reg_type.upper()} Val', 
             color=colors[reg_type], 
             linestyle='--')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 绘制准确率曲线
plt.subplot(1, 2, 2)
for reg_type in histories:
    plt.plot(histories[reg_type]['train_acc'], 
             label=f'{reg_type.upper()} Train', 
             color=colors[reg_type], 
             linestyle='-')
    plt.plot(histories[reg_type]['val_acc'], 
             label=f'{reg_type.upper()} Val', 
             color=colors[reg_type], 
             linestyle='--')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
plt.show()

# 绘制最终准确率对比
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color=[colors[k] for k in results.keys()])
plt.title('Test Accuracy Comparison')
plt.xlabel('Regularization Type')
plt.ylabel('Accuracy (%)')
plt.ylim(98, 99.5)
for i, v in enumerate(results.values()):
    plt.text(i, v+0.1, f"{v:.2f}%", ha='center')
plt.savefig('accuracy_comparison.png', dpi=300)
plt.show()

# 打印最终结果
print("\n=== Final Test Accuracy ===")
for reg_type, acc in results.items():
    print(f"{reg_type.upper():<6}: {acc:.2f}%")

# 可视化权重分布
plt.figure(figsize=(15, 4))
for i, reg_type in enumerate(['none', 'l1', 'l2']):
    model = CNN().to(device)
    model.load_state_dict(torch.load(f'best_model_{reg_type}.pth'))
    
    # 获取所有权重
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.extend(param.data.cpu().numpy().flatten())
    
    plt.subplot(1, 3, i+1)
    plt.hist(weights, bins=100, range=(-0.5, 0.5), color=colors[reg_type])
    plt.title(f'{reg_type.upper()} Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('weight_distributions.png', dpi=300)
plt.show()