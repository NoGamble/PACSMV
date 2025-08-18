import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table

# 准备数据（移除了Test行）
data = [
    ["Epoch 1", 0.2495, 92.39, 0.0732, 97.79],
    ["Epoch 8 (Best Val)", 0.0407, 98.77, 0.0310, 99.17],
    ["Epoch 11 (Early Stop)", 0.0337, 98.89, 0.0339, 99.09]
]
columns = ["Stage", "Train Loss", "Train Acc", "Val Loss", "Val Acc"]
df = pd.DataFrame(data, columns=columns)

# 创建图形
plt.figure(figsize=(10, 2.5))  # 减小高度
ax = plt.gca()
ax.axis('off')

# 自定义单元格颜色（3行交替）
cell_colors = [
    ['#F5F5F5']*5,  # 第一行底色
    ['#E6F3FF']*5,  # 第二行底色
    ['#FFE6E6']*5   # 第三行底色
]

# 创建表格
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center',
    colColours=['#404040']*5,
    cellColours=cell_colors
)

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)  # 调整比例

# 高亮关键数据
for (row, col), cell in table.get_celld().items():
    if row == 0:  # 标题行
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#404040')
    elif row == 2 and col == 3:  # 最佳验证损失
        cell.set_text_props(color='red', weight='bold')
    elif row == 2 and col == 4:  # 最佳验证准确率
        cell.set_text_props(color='green', weight='bold')

# 添加标题
plt.title("Training Metrics", y=1.1, fontsize=14, fontweight='bold')
plt.tight_layout()

# 保存和显示
plt.savefig('training_metrics_no_test.png', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white')
plt.show()