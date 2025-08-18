"""
MNIST Handwritten Digit Classification - Logistic Regression with Multiple Optimizers
Enhanced Visualization with Seaborn
Author: Modified based on Yujing Xiong's original code
"""

import numpy as np
import time 
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay,
                             precision_score, recall_score, f1_score)
from sklearn.decomposition import PCA

# Set global style
sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams['figure.dpi'] = 120
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def load_and_preprocess_data():
    """Load and preprocess MNIST data with optional PCA"""
    print("=== Loading MNIST Dataset ===")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data.astype('float32'), mnist.target.astype('int')
    
    # Split into train(60%), validation(20%), test(20%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA dimensionality reduction
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"Original dimension: {X_train.shape[1]}")
    print(f"Reduced dimension after PCA: {X_train_pca.shape[1]}")
    return X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, pca

def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train and evaluate logistic regression with different optimizers"""
    solvers = {
        'lbfgs': {
            'solver': 'lbfgs',
            'max_iter': 500,
            'random_state': 42
        },
        'saga_l1': {
            'solver': 'saga',
            'max_iter': 500,
            'penalty': 'l1',
            'C': 0.1,
            'random_state': 42
        },
        'saga_l2': {
            'solver': 'saga',
            'max_iter': 500,
            'penalty': 'l2',
            'random_state': 42
        },
        'newton-cg': {
            'solver': 'newton-cg',
            'max_iter': 500,
            'random_state': 42
        }
    }
    
    results = {}
    
    for name, params in solvers.items():
        print(f"\n=== Training {name} optimizer ===")
        start_time = time.time()
        
        model = LogisticRegression(**params, n_jobs=-1)
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # Calculate extended metrics
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        results[name] = {
            'model': model,
            'train_time': train_time,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'classification_report': classification_report(y_test, y_test_pred, output_dict=True)
        }
        
        print(f"Training time: {train_time:.2f}s")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test F1-score: {test_f1:.4f}")
    
    return results
def visualize_results(results):
    # Prepare metrics dataframe
    metrics = []
    for opt in results:
        # Extract classification report data
        report = results[opt]['classification_report']
        for label in range(10):  # For each digit class
            metrics.append({
                'Optimizer': opt,
                'Class': str(label),
                'Precision': report[str(label)]['precision'],
                'Recall': report[str(label)]['recall'],
                'F1': report[str(label)]['f1-score']
            })
        # Add weighted averages
        metrics.append({
            'Optimizer': opt,
            'Class': 'Weighted Avg',
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1': report['weighted avg']['f1-score']
        })
    
    df = pd.DataFrame(metrics)
    
    # Create figure
    plt.figure(figsize=(18, 10))
    plt.suptitle('Model Performance Metrics Comparison', y=1.02, fontsize=16)
    
    # 1. F1 Score Comparison (Main Plot)
    plt.subplot(2, 2, 1)
    sns.barplot(data=df[df['Class'] == 'Weighted Avg'], 
                x='Optimizer', y='F1', palette="viridis")
    plt.title('Weighted F1 Score Comparison')
    plt.ylabel('F1 Score')
    plt.ylim(0.8, 1.0)
    # Add value labels
    for p in plt.gca().patches:
        plt.gca().annotate(f"{p.get_height():.4f}", 
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center', xytext=(0, 10), 
                          textcoords='offset points')
    
    # 2. Precision-Recall Tradeoff
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df[df['Class'] == 'Weighted Avg'], 
                   x='Recall', y='Precision', hue='Optimizer',
                   s=200, palette="plasma")
    plt.title('Precision-Recall Tradeoff')
    plt.xlim(0.85, 1.0)
    plt.ylim(0.85, 1.0)
    # Add optimizer labels
    for line in range(df[df['Class'] == 'Weighted Avg'].shape[0]):
        plt.text(df[df['Class'] == 'Weighted Avg']['Recall'].iloc[line]+0.002, 
                df[df['Class'] == 'Weighted Avg']['Precision'].iloc[line], 
                df[df['Class'] == 'Weighted Avg']['Optimizer'].iloc[line],
                horizontalalignment='left', size='medium')
    
    # 3. Class-wise F1 Scores
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df[df['Class'] != 'Weighted Avg'], 
               x='Optimizer', y='F1', palette="coolwarm")
    plt.title('Class-wise F1 Score Distribution')
    plt.ylabel('F1 Score')
    plt.xlabel('Optimizer')
    
    # 4. Metrics Radar Chart (Alternative View)
    plt.subplot(2, 2, 4, polar=True)
    
    # Prepare data for radar chart
    metrics = ['Precision', 'Recall', 'F1']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    for opt in results:
        values = df[(df['Optimizer'] == opt) & (df['Class'] == 'Weighted Avg')][metrics].values[0].tolist()
        values += values[:1]  # Close the loop
        plt.plot(angles, values, linewidth=2, 
                label=opt, marker='o')
        plt.fill(angles, values, alpha=0.1)
    
    # Format radar chart
    plt.title('Metrics Comparison (Weighted Avg)', y=1.1)
    plt.xticks(angles[:-1], metrics)
    plt.yticks([0.85, 0.90, 0.95, 1.0], ["0.85", "0.90", "0.95", "1.0"])
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, pca = load_and_preprocess_data()
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Visualize results
    visualize_results(results, y_test)
    
    # Print best model summary
    best_name, best_result = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print("\n=== BEST MODEL SUMMARY ===")
    print(f"Optimizer: {best_name}")
    print(f"Test Accuracy: {best_result['test_accuracy']:.4f}")
    print(f"Test F1-score: {best_result['test_f1']:.4f}")
    print(f"Training Time: {best_result['train_time']:.2f}s")

if __name__ == '__main__':
    main()