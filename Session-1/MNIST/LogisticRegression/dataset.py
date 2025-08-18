import matplotlib.pyplot as plt

# Data splits
labels = ['Train (60%)', 'Validation (20%)', 'Test (20%)']
sizes = [60, 20, 20]
colors = ['#66b3ff', '#99ff99', '#ffcc99']  # 蓝, 绿, 橙

# Plot
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('MNIST Dataset Split', fontsize=14)
plt.axis('equal')  # 确保饼图是圆形
plt.savefig('dataset_split_pie.png', dpi=300, bbox_inches='tight')
plt.show()