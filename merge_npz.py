import numpy as np

def merge_npz_files(original_file, append_file, output_file):
    print(f"正在读取原数据: {original_file}")
    # 注意：包含字典对象的 numpy 数组必须加 allow_pickle=True
    orig_data = np.load(original_file, allow_pickle=True)['datas']
    
    print(f"正在读取要追加的数据: {append_file}")
    new_data = np.load(append_file, allow_pickle=True)['datas']
    
    print(f"原数据段数: {len(orig_data)}")
    print(f"追加数据段数: {len(new_data)}")
    
    # 将两个数组拼接
    merged_data = np.concatenate((orig_data, new_data))
    
    print(f"合并后总数据段数: {len(merged_data)}")
    
    # 保存为新的 npz 文件
    np.savez_compressed(output_file, datas=merged_data)
    print(f"✅ 合并成功！新文件已保存至: {output_file}")

if __name__ == "__main__":
    # 文件路径配置 (请根据实际情况修改)
    original_npz = "koopman_train.npz"           # 原来已有的 7小时切出来的训练集
    append_npz = "koopman_train_left_turn.npz"   # 第一步刚提取出的 左转差速训练集
    output_merged_npz = "koopman_train_merged.npz" # 最终合并输出的文件名
    
    merge_npz_files(original_npz, append_npz, output_merged_npz)