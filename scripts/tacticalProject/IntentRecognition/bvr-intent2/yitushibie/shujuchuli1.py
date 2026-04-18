import pandas as pd
import numpy as np
import os
import glob

# 定义意图映射
intent_mapping = {
    'lock_on': 0,
    'attack_maneuver': 0,
    'search': 1,
    'formation_maintain': 2,
    'coordination': 2,
    'defensive_maneuver': 3,
    'evasive_maneuver': 3,
    'escape': 4,
    'unknown': 5
}

# 定义独热编码映射
one_hot_mapping = {
    0: [1, 0, 0, 0, 0, 0],  # 攻击
    1: [0, 1, 0, 0, 0, 0],  # 侦察
    2: [0, 0, 1, 0, 0, 0],  # 协同
    3: [0, 0, 0, 1, 0, 0],  # 防御
    4: [0, 0, 0, 0, 1, 0],  # 逃逸
    5: [0, 0, 0, 0, 0, 1]   # 未知
}

# 输入文件夹列表
input_folders = [
    'drag_shoot',
    'front_back', 
    'high_low',
    'pincer_attack',
    'side_by_side'
]

# 输出文件夹
output_folder = 'chuli1'
os.makedirs(output_folder, exist_ok=True)

def process_file(file_path, output_path):
    """处理单个CSV文件"""
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 处理敌方两机的动作列
        for col in ['action_B0100', 'action_B0200']:
            if col in df.columns:
                # 映射动作到意图
                intent_col = col.replace('action', 'intent')
                df[intent_col] = df[col].map(intent_mapping).fillna(5)  # 默认未知
                
                # 生成独热编码列
                for i in range(6):
                    one_hot_col = f"{col.replace('action', 'one_hot')}_{i}"
                    df[one_hot_col] = df[intent_col].apply(lambda x: one_hot_mapping[x][i])
        
        # 保存处理后的文件
        df.to_csv(output_path, index=False)
        print(f"成功处理: {os.path.basename(file_path)}")
        return True
    except Exception as e:
        print(f"处理文件时出错 {os.path.basename(file_path)}: {str(e)}")
        return False

def main():
    """主函数"""
    total_files = 0
    success_files = 0
    
    # 遍历所有输入文件夹
    for folder in input_folders:
        folder_path = os.path.join(os.getcwd(), folder)
        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            continue
        
        # 创建对应的输出子文件夹
        output_subfolder = os.path.join(output_folder, folder)
        os.makedirs(output_subfolder, exist_ok=True)
        
        # 查找所有CSV文件
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        
        # 处理每个CSV文件
        for csv_file in csv_files:
            total_files += 1
            file_name = os.path.basename(csv_file)
            output_path = os.path.join(output_subfolder, file_name)
            
            if process_file(csv_file, output_path):
                success_files += 1
    
    print(f"\n处理完成!")
    print(f"总文件数: {total_files}")
    print(f"成功处理: {success_files}")
    print(f"失败处理: {total_files - success_files}")
    print(f"结果保存在: {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    main()