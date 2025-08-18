import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

def draw_mlp_5layers_explicit():
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    
    # 明确定义五层（含Dropout作为独立层）
    layers = [
        {"type": "Flatten", "size": "28×28→784", "color": "#a6cee3", "width": 1.2},
        {"type": "Linear+ReLU", "size": "512", "color": "#1f78b4", "width": 1.5},
        {"type": "Dropout", "size": "p=0.1", "color": "#ffff99", "width": 1.0},
        {"type": "Linear+ReLU", "size": "256", "color": "#33a02c", "width": 1.3},
        {"type": "Output", "size": "10", "color": "#e31a1c", "width": 0.8}
    ]
    
    # 绘制层
    for i, layer in enumerate(layers):
        # 主矩形
        rect = Rectangle((i*3, 0), layer["width"], 3, 
                        facecolor=layer["color"], alpha=0.7, edgecolor="k")
        ax.add_patch(rect)
        
        # 层类型标注
        plt.text(i*3 + layer["width"]/2, 3.2, layer["type"], 
                ha="center", va="center", fontsize=11, weight="bold")
        
        # 层参数标注
        plt.text(i*3 + layer["width"]/2, 1.5, layer["size"], 
                ha="center", va="center", fontsize=10)
    
    # 绘制连接箭头
    for i in range(len(layers)-1):
        arrow = FancyArrow(
            i*3 + layers[i]["width"], 1.5,
            2.8 - layers[i]["width"], 0, width=0.15,
            head_width=0.3, head_length=0.3,
            fc="#7f7f7f", ec="#7f7f7f"
        )
        ax.add_patch(arrow)
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Flatten (No params)',
                  markerfacecolor='#a6cee3', markersize=15),
        plt.Line2D([0], [0], marker='s', color='w', label='Linear+ReLU (Trainable)',
                  markerfacecolor='#1f78b4', markersize=15),
        plt.Line2D([0], [0], marker='s', color='w', label='Dropout (Regularization)',
                  markerfacecolor='#ffff99', markersize=15),
        plt.Line2D([0], [0], marker='s', color='w', label='Output Layer',
                  markerfacecolor='#e31a1c', markersize=15)
    ]
    plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.35, 1))
    
    plt.xlim(-1, len(layers)*3)
    plt.ylim(-1, 4)
    plt.axis("off")
    plt.title("MLP Architecture with Explicit 5 Layers (Including Dropout)", 
             fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig("mlp_5layers_explicit.png", dpi=300, bbox_inches="tight")
    plt.show()

draw_mlp_5layers_explicit()