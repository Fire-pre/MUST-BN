import pandas as pd
import numpy as np

def clean_flight_data(file_path, output_path):
    print("开始数据清洗...")
    # 1. 加载数据
    df = pd.read_csv(file_path)
    
    # 2. 移除论文中提到的不相关列 (Section 3.3)
    # 根据论文描述，YEAR, FL_DATE 以及冗余的机场/地点信息需要剔除
    cols_to_drop = [
        'YEAR', 'FL_DATE', 'ORIGIN_CITY_MARKET_ID', 'DEST_CITY_MARKET_ID',
        'ARR_DEL15', 'DEP_DEL15', 'DEP_DELAY_GROUP', 'ARR_DELAY_GROUP',
        'FLIGHTS', 'CANCELLED', 'DIV_AIRPORT_LANDINGS'
    ]
    # 仅删除存在的列，避免报错
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    # 3. 处理延误原因列的缺失值 (Section 3.3)
    # 论文指出：没有延误时数据可能缺失，应填充为 0
    delay_cause_cols = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY']
    for col in delay_cause_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 4. 删除其他包含缺失值的行
    initial_rows = len(df)
    df.dropna(inplace=True)
    print(f"删除了 {initial_rows - len(df)} 行包含缺失值的无效数据。")

    # 5. 生成目标变量 Status (Section 3.1 & Table 2)
    # 论文定义：ARR_DELAY <= 15 为 1 (on-time), > 15 为 0 (delayed)
    if 'ARR_DELAY' in df.columns:
        df['Status'] = (df['ARR_DELAY'] <= 15).astype(int)
    
    # 保存中间结果
    df.to_csv(output_path, index=False)
    print(f"清洗完成，已保存至: {output_path}")

if __name__ == "__main__":
    # 请将 'your_data.csv' 替换为你实际的文件名
    clean_flight_data('T_ONTIME_REPORTING.csv', 'cleaned_flight_data.csv')