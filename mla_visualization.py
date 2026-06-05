#!/usr/bin/env python3
"""
MLA (Multi-head Latent Attention) 性能可视化分析
生成展示MLA优势的图表
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def plot_memory_comparison():
    """绘制内存占用对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 数据准备
    seq_lengths = [4, 16, 32, 128]  # K tokens
    traditional_mha = [15.6, 62.4, 124.8, 499.2]  # GB
    mla = [1.95, 7.8, 15.6, 62.4]  # GB
    
    # 柱状图对比
    x = np.arange(len(seq_lengths))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, traditional_mha, width, label='Traditional MHA', color='#ff7f0e')
    bars2 = ax1.bar(x + width/2, mla, width, label='MLA', color='#2ca02c')
    
    ax1.set_xlabel('Sequence Length (K tokens)')
    ax1.set_ylabel('KV Cache Size (GB)')
    ax1.set_title('KV Cache Memory Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{s}K' for s in seq_lengths])
    ax1.legend()
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # 压缩率展示
    compression_rates = [(t - m) / t * 100 for t, m in zip(traditional_mha, mla)]
    ax2.plot(seq_lengths, compression_rates, 'o-', color='#d62728', linewidth=2, markersize=8)
    ax2.fill_between(seq_lengths, compression_rates, alpha=0.3, color='#d62728')
    ax2.set_xlabel('Sequence Length (K tokens)')
    ax2.set_ylabel('Memory Reduction (%)')
    ax2.set_title('Memory Compression Rate')
    ax2.set_ylim(85, 90)
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for x, y in zip(seq_lengths, compression_rates):
        ax2.annotate(f'{y:.1f}%',
                    xy=(x, y),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/workspace/mla_memory_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_metrics():
    """绘制性能指标对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 数据准备
    metrics = ['Inference Speed', 'Memory Usage', 'Bandwidth Requirement', 'GPU Utilization']
    traditional = [1, 100, 100, 60]  # 标准化为百分比
    mla = [5.7, 6.7, 15, 95]  # MLA相对值
    
    # 将传统MHA标准化为100%，MLA相应调整
    mla_normalized = [570, 6.7, 15, 158.3]  # 相对于传统MHA的百分比
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, traditional, width, label='Traditional MHA', color='#1f77b4')
    bars2 = ax.bar(x + width/2, mla_normalized, width, label='MLA', color='#ff7f0e')
    
    ax.set_ylabel('Relative Performance (%)')
    ax.set_title('MLA vs Traditional MHA Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 600)
    
    # 添加改进倍数标签
    for i, (t, m) in enumerate(zip(traditional, mla_normalized)):
        if m > t:
            improvement = f'{m/t:.1f}x'
            color = 'green'
        else:
            improvement = f'{t/m:.1f}x reduction'
            color = 'red'
        
        ax.text(i, max(t, m) + 10, improvement, 
                ha='center', va='bottom', color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/workspace/mla_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_latency_comparison():
    """绘制不同应用场景的延迟对比"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 数据准备
    applications = ['Dialog Generation\n(8K tokens)', 
                   'Document Summary\n(32K tokens)', 
                   'Code Generation\n(16K tokens)']
    traditional_latency = [450, 2100, 890]  # ms
    mla_latency = [125, 580, 215]  # ms
    
    x = np.arange(len(applications))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, traditional_latency, width, 
                    label='Traditional MHA', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, mla_latency, width, 
                    label='MLA', color='#27ae60', alpha=0.8)
    
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Inference Latency Comparison Across Applications')
    ax.set_xticks(x)
    ax.set_xticklabels(applications)
    ax.legend()
    
    # 添加数值和加速比
    for i, (t, m) in enumerate(zip(traditional_latency, mla_latency)):
        # 传统MHA数值
        ax.text(i - width/2, t + 20, f'{t}ms', 
                ha='center', va='bottom', fontsize=9)
        # MLA数值
        ax.text(i + width/2, m + 20, f'{m}ms', 
                ha='center', va='bottom', fontsize=9)
        # 加速比
        speedup = t / m
        ax.text(i, max(t, m) + 100, f'{speedup:.1f}x faster', 
                ha='center', va='bottom', color='darkgreen', 
                fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('/workspace/mla_latency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_compression_visualization():
    """可视化MLA的压缩原理"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建示意图
    # 原始KV矩阵
    original_width = 8
    original_height = 4
    compressed_width = 1
    
    # 绘制原始KV矩阵
    rect1 = plt.Rectangle((1, 5), original_width, original_height, 
                         facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(rect1)
    ax.text(5, 7, 'Original K,V\n(d_model × seq_len)\n~100GB', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 绘制压缩过程箭头
    ax.arrow(9.5, 7, 1.5, 0, head_width=0.3, head_length=0.2, 
             fc='black', ec='black', linewidth=2)
    ax.text(10.25, 7.5, 'Low-rank\nProjection', ha='center', va='bottom', fontsize=10)
    
    # 绘制潜在向量
    rect2 = plt.Rectangle((12, 5), compressed_width, original_height, 
                         facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(rect2)
    ax.text(12.5, 7, 'Latent\nVector\n(d_c × seq_len)\n~6.7GB', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 绘制重建过程
    ax.arrow(13.5, 7, 1.5, 0, head_width=0.3, head_length=0.2, 
             fc='black', ec='black', linewidth=2)
    ax.text(14.25, 7.5, 'Reconstruction\nfor each head', ha='center', va='bottom', fontsize=10)
    
    # 绘制重建的KV
    for i in range(3):
        rect3 = plt.Rectangle((16, 3 + i*2), original_width*0.3, original_height*0.3, 
                             facecolor='#2ecc71', edgecolor='black', linewidth=1)
        ax.add_patch(rect3)
    ax.text(17.2, 7, 'Reconstructed\nK¹,V¹\n...\nKⁿ,Vⁿ', 
            ha='center', va='center', fontsize=10)
    
    # 添加压缩率标注
    ax.text(7, 2, '93.3% Compression Rate', 
            ha='center', va='center', fontsize=16, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))
    
    # 设置坐标轴
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('MLA Compression Mechanism', fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('/workspace/mla_compression_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """生成所有可视化图表"""
    print("Generating MLA visualization charts...")
    
    plot_memory_comparison()
    print("✓ Memory comparison chart generated")
    
    plot_performance_metrics()
    print("✓ Performance metrics chart generated")
    
    plot_latency_comparison()
    print("✓ Latency comparison chart generated")
    
    plot_compression_visualization()
    print("✓ Compression mechanism visualization generated")
    
    print("\nAll charts have been saved to /workspace/")

if __name__ == "__main__":
    main()