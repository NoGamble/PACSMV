# 完整代码：MNIST数据加载、PCA降维及可视化（修正版）
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. 加载并预处理数据
def load_and_preprocess():
    print("=== Loading MNIST Dataset ===")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data.astype('float32'), mnist.target.astype('int')
    
    # 数据分割 (60%训练, 20%验证, 20%测试)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA降维 (保留95%方差)
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"Original dimension: {X_train.shape[1]}")
    print(f"Reduced dimension after PCA: {X_train_pca.shape[1]}")
    return X_train, X_train_scaled, X_train_pca, y_train, pca

# 2. 可视化函数
def visualize_pca_results(X_original, X_scaled, pca, X_pca, y, n_samples=1000):
    plt.figure(figsize=(18, 12))
    sns.set_style("whitegrid")
    plt.suptitle("MNIST PCA Analysis Results", fontsize=16, y=1.02)
    
    # 2.1 方差解释比例曲线
    plt.subplot(2, 2, 1)
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    sns.lineplot(x=np.arange(1, len(cumulative_var)+1), 
                 y=cumulative_var,
                 color='royalblue', linewidth=2.5)
    plt.axhline(y=0.95, color='crimson', linestyle='--', 
                label='95% Variance Threshold')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance Ratio')
    plt.legend()
    
    # 2.2 前两个主成分散点图
    plt.subplot(2, 2, 2)
    sample_idx = np.random.choice(len(X_pca), n_samples, replace=False)
    sns.scatterplot(x=X_pca[sample_idx, 0], y=X_pca[sample_idx, 1],
                    hue=y[sample_idx], palette='tab10',
                    alpha=0.7, s=50, edgecolor='none')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Var)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Var)')
    plt.title('2D Projection of MNIST Data')
    plt.legend(title='Digit', bbox_to_anchor=(1.05, 1))
    
    # 2.3 主成分热力图 (前10个PC)
    plt.subplot(2, 2, 3)
    component_matrix = pca.components_[:10]
    sns.heatmap(component_matrix, cmap='coolwarm',
                xticklabels=False, 
                yticklabels=[f'PC{i+1}' for i in range(10)],
                cbar_kws={'label': 'Feature Weight'})
    plt.title('Top 10 Principal Components')
    plt.ylabel('Principal Components')
    plt.xlabel('Original Pixel Features')
    
    # 2.4 数字重建示例（修正版）
    plt.subplot(2, 2, 4)
    example_idx = np.random.randint(0, len(X_pca))
    
    # 获取三种状态的图像
    original_img = X_original[example_idx].reshape(28, 28)          # 原始[0,255]
    scaled_img = X_scaled[example_idx].reshape(28, 28)                # 标准化后（含负值）
    reconstructed_img = pca.inverse_transform(X_pca[example_idx]).reshape(28, 28)  # 重建值
    
    # 关键修改：对每幅图像进行适当的归一化处理
    display_images = [
        original_img / 255.0,  # 原始图像归一化到[0,1]
        (scaled_img - scaled_img.min()) / (scaled_img.max() - scaled_img.min()),  # 标准化图像归一化
        (reconstructed_img - reconstructed_img.min()) / 
        (reconstructed_img.max() - reconstructed_img.min())  # 重建图像归一化
    ]
    
    # 显示三幅图像对比
    plt.imshow(np.hstack(display_images), 
               cmap='gray')
    plt.title('Original (L) | Scaled (M) | Reconstructed (R)\nDigit: {}'.format(y[example_idx]))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# 3. 主程序
if __name__ == '__main__':
    # 加载数据并降维
    X_train, X_train_scaled, X_train_pca, y_train, pca = load_and_preprocess()
    
    # 可视化结果
    visualize_pca_results(X_train, X_train_scaled, pca, X_train_pca, y_train)