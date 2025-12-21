import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置绘图风格，适合学术报告
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk") 

def generate_visuals(input_path):
    df = pd.read_csv(input_path)
    
    # --- 1. 相关性热力图 (展现 R 公式的作用) ---
    plt.figure(figsize=(12, 10))
    # 选择 Table 2 中的核心数值特征进行展示
    core_cols = ['MONTH', 'DAY_OF_MONTH', 'CRS_DEP_TIME', 'DEP_DELAY', 'ARR_DELAY', 'CARRIER_DELAY', 'NAS_DELAY']
    corr = df[core_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool)) # 只看下半部分
    sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', fmt=".2f", center=0)
    plt.title('Feature Correlation Matrix (Pearson R)', fontsize=18)
    plt.tight_layout()
    plt.savefig('visual_1_correlation_heatmap.png', dpi=300)
    plt.show()

    # --- 2. Box-Cox 变换前后对比 (展现 Q 公式的作用) ---
    # 以 DEP_DELAY 为例，这是航空数据中最典型的偏态数据
    plt.figure(figsize=(14, 6))
    
    # 变换前
    plt.subplot(1, 2, 1)
    sns.histplot(df['DEP_DELAY'], kde=True, color='skyblue')
    plt.title('Original DEP_DELAY (Skewed)', fontsize=15)
    plt.xlabel('Delay Minutes')

    # 执行 Box-Cox (先平移确保为正)
    shift_val = abs(df['DEP_DELAY'].min()) + 1
    transformed_data, _ = stats.boxcox(df['DEP_DELAY'] + shift_val)
    
    # 变换后
    plt.subplot(1, 2, 2)
    sns.histplot(transformed_data, kde=True, color='salmon')
    plt.title('After Box-Cox Transformation (Normal-like)', fontsize=15)
    plt.xlabel('Transformed Value')
    
    plt.tight_layout()
    plt.savefig('visual_2_boxcox_comparison.png', dpi=300)
    plt.show()

    # --- 3. 目标变量 Status 分布 (展示类别不平衡) ---
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Status', data=df, palette='viridis')
    plt.title('Flight Status Distribution (0: Delayed, 1: On-time)', fontsize=15)
    plt.xticks([0, 1], ['Delayed (>15min)', 'On-time'])
    plt.tight_layout()
    plt.savefig('visual_3_status_distribution.png', dpi=300)
    plt.show()

    # --- 4. 关键特征与延迟的散点回归 (证明 R 的线性趋势) ---
    plt.figure(figsize=(10, 6))
    # 展示出发延迟与到达延迟的强相关性
    sns.regplot(x='DEP_DELAY', y='ARR_DELAY', data=df.sample(2000), 
                scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'})
    plt.title('DEP_DELAY vs ARR_DELAY (Regression Analysis)', fontsize=15)
    plt.tight_layout()
    plt.savefig('visual_4_regression_plot.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # 使用上一步清洗后的数据
    generate_visuals('cleaned_flight_data.csv')