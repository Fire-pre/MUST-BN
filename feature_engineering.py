import pandas as pd
import numpy as np
from scipy import stats

def feature_engineering(input_path, final_output_path):
    print("开始特征筛选与 Box-Cox 变换...")
    df = pd.read_csv(input_path)

    # 1. 特征筛选 (Table 2: Selected Features)
    # 按照论文 Table 2 选定的核心列
    selected_features = [
        'MONTH', 'DAY_OF_MONTH', 'OP_CARRIER_AIRLINE_ID', 'OP_CARRIER_FL_NUM',
        'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'CRS_DEP_TIME', 'DEP_TIME', 
        'DEP_DELAY', 'CRS_ARR_TIME', 'ARR_DELAY', 'CARRIER_DELAY', 
        'SECURITY_DELAY', 'Status'
    ]
    
    # 仅保留数据集中存在的选定列
    available_features = [f for f in selected_features if f in df.columns]
    df = df[available_features]

    # 2. 相关性分析 R (Section 3.4)
    # 计算皮尔逊相关系数矩阵
    corr_matrix = df.corr()
    print("\n特征与 ARR_DELAY 的相关性 (R):")
    if 'ARR_DELAY' in corr_matrix.columns:
        print(corr_matrix['ARR_DELAY'].sort_values(ascending=False))

    # 3. Box-Cox 变换 Q (Section 3.5)
    # 论文要求将非正态数据正态化。Box-Cox 要求输入必须 > 0。
    # 我们对连续数值列进行变换：DEP_DELAY, ARR_DELAY, CARRIER_DELAY
    numeric_to_transform = ['DEP_DELAY', 'ARR_DELAY', 'CARRIER_DELAY']
    
    for col in numeric_to_transform:
        if col in df.columns:
            # 数据平移：因为延误可能有负数或0，加一个常数使其全部为正
            shift_value = abs(df[col].min()) + 1
            data_to_transform = df[col] + shift_value
            
            # 执行 Box-Cox 变换 (Q 公式)
            # stats.boxcox 会自动寻找最优的 alpha
            transformed_data, best_alpha = stats.boxcox(data_to_transform)
            df[f'{col}_transformed'] = transformed_data
            print(f"已对 {col} 完成 Box-Cox 变换，最优 alpha: {best_alpha:.4f}")

    # 4. 保存最终用于模型训练的特征集
    df.to_csv(final_output_path, index=False)
    print(f"\n特征工程完成！最终数据集维度: {df.shape}")
    print(f"文件已保存至: {final_output_path}")

if __name__ == "__main__":
    feature_engineering('cleaned_flight_data.csv', 'final_features_for_model.csv')