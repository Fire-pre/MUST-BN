import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.style.use('seaborn-v0_8-whitegrid')

def generate_final_presentation_visuals(input_path):
    df = pd.read_csv(input_path)
    
    # --- 1. 热力图 (已很完美，仅微调颜色使对比更鲜明) ---
    plt.figure(figsize=(10, 8))
    core_cols = ['DAY_OF_MONTH', 'CRS_DEP_TIME', 'DEP_DELAY', 'ARR_DELAY', 'CARRIER_DELAY']
    corr = df[core_cols].corr()
    sns.heatmap(corr, annot=True, cmap='RdYlBu_r', fmt=".2f", center=0, annot_kws={"size": 14})
    plt.title('Feature Correlation Matrix (Pearson R)', fontsize=16, pad=20)
    plt.savefig('Final_Heatmap.png', dpi=300)

    # --- 2. Box-Cox 对比图 (大幅优化右图展示效果) ---
    plt.figure(figsize=(14, 6))
    
    # 左图：原始分布
    plt.subplot(1, 2, 1)
    sns.histplot(df['DEP_DELAY'], kde=True, color='#3498db', bins=50)
    plt.xlim(-20, 200) 
    plt.title('Before: Original DEP_DELAY (Skewed)', fontsize=14)
    
    # 右图：变换后分布
    plt.subplot(1, 2, 2)
    shift_val = abs(df['DEP_DELAY'].min()) + 1
    transformed_data, _ = stats.boxcox(df['DEP_DELAY'] + shift_val)
    # 关键：手动设置右图的 X 轴范围，展示“钟形”
    sns.histplot(transformed_data, kde=True, color='#e74c3c', bins=50)
    mean_val = transformed_data.mean()
    std_val = transformed_data.std()
    plt.xlim(mean_val - 4*std_val, mean_val + 4*std_val) # 只展示均值附近 4 个标准差的范围
    plt.title('After: Box-Cox Transformed (Normalized)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('Final_BoxCox_Comparison.png', dpi=300)

    # --- 3. 回归图 (增加 R 值注释) ---
    plt.figure(figsize=(9, 7))
    sample_df = df.sample(min(3000, len(df)))
    r_value = df['DEP_DELAY'].corr(df['ARR_DELAY']) # 计算 R
    
    g = sns.regplot(x='DEP_DELAY', y='ARR_DELAY', data=sample_df, 
                scatter_kws={'alpha':0.2, 's':15, 'color':'#2c3e50'}, 
                line_kws={'color':'#e67e22', 'lw':3})
    
    # 在图上添加 R 值文本提示
    plt.text(50, 900, f'Pearson R = {r_value:.2f}', fontsize=16, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
    
    plt.xlim(-50, 1100)
    plt.ylim(-50, 1100)
    plt.title('Linear Relationship Validation', fontsize=16)
    plt.savefig('Final_Regression_Analysis.png', dpi=300)
    
    print("最终版 PPT 图片已生成！")

if __name__ == "__main__":
    generate_final_presentation_visuals('final_features_for_model.csv')