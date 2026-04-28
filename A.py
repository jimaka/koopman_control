import numpy as np
import yaml
import os

def convert_npy_to_yaml(npy_files_dict, output_yaml="config.yaml"):
    """
    Args:
        npy_files_dict: 字典格式，Key 为 YAML 里的变量名，Value 为 .npy 文件路径
        output_yaml: 输出文件名
    """
    combined_data = {}

    for key, file_path in npy_files_dict.items():
        if os.path.exists(file_path):
            # 加载 npy 数据
            data = np.load(file_path)
            
            # YAML 不支持 numpy 数组类型，必须转换为 list
            # 如果是多维矩阵，tolist() 会保持嵌套结构
            combined_data[key] = data.tolist()
            print(f"已加载 {file_path}, 形状为: {data.shape}")
        else:
            print(f"警告: 文件 {file_path} 不存在，跳过。")

    # 写入 YAML 文件
    with open(output_yaml, 'w', encoding='utf-8') as f:
        # default_flow_style=None 会生成更易读的矩阵格式
        yaml.dump(combined_data, f, default_flow_style=None, sort_keys=False)
    
    print(f"\n成功！所有数据已保存至: {output_yaml}")

if __name__ == "__main__":
    # 配置你需要读取的文件映射
    # 假设你之前导出了这些文件
    files_to_load = {
        "A_matrix": "A.npy",
        "B_matrix": "B.npy",
        "C_matrix": "C.npy"
    }

    convert_npy_to_yaml(files_to_load, "koopman_params.yaml")