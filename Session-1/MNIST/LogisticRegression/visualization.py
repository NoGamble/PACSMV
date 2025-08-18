import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 准备数据
data = {
    'Optimizer': ['lbfgs', 'saga_l1', 'saga_l2', 'newton-cg'],
    'Training Time (s)': [7.54, 259.71, 174.50, 25.10],
    'Validation Accuracy': [0.9210, 0.9191, 0.9197, 0.9208],
    'Test Accuracy': [0.9189, 0.9199, 0.9197, 0.9185]
}
df = pd.DataFrame(data)

# 设置Seaborn风格
sns.set_style("whitegrid")
plt.figure(figsize=(14, 10))
plt.suptitle('MNIST Classification Performance Comparison', y=1.02, fontsize=16)

# 1. 准确率对比图
plt.subplot(2, 2, 1)
ax1 = sns.barplot(x='Optimizer', y='Test Accuracy', data=df, palette="Blues_d")
ax1.set_title('Test Accuracy Comparison')
ax1.set_ylim(0.915, 0.922)
for p in ax1.patches:
    ax1.annotate(f'{p.get_height():.4f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points')

# 2. 训练时间对比图
plt.subplot(2, 2, 2)
ax2 = sns.barplot(x='Optimizer', y='Training Time (s)', data=df, palette="Reds_d")
ax2.set_title('Training Time Comparison')
ax2.set_yscale('log')  # 对数坐标显示时间差异
for p in ax2.patches:
    ax2.annotate(f'{p.get_height():.1f}s', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points')

# 3. 准确率-时间效率散点图
plt.subplot(2, 2, 3)
ax3 = sns.scatterplot(x='Training Time (s)', y='Test Accuracy', 
                      hue='Optimizer', s=200, data=df)
ax3.set_title('Accuracy vs Training Time')
ax3.set_xscale('log')
ax3.grid(True, which="both", ls="--")

# 4. 验证集与测试集准确率对比
plt.subplot(2, 2, 4)
df_melted = df.melt(id_vars=['Optimizer'], 
                    value_vars=['Validation Accuracy', 'Test Accuracy'],
                    var_name='Dataset', value_name='Accuracy')
ax4 = sns.barplot(x='Optimizer', y='Accuracy', hue='Dataset', data=df_melted)
ax4.set_title('Validation vs Test Accuracy')
ax4.set_ylim(0.915, 0.922)

plt.tight_layout()
plt.savefig('mnist_seaborn_results.png', dpi=300, bbox_inches='tight')
plt.show()