import os
import numpy as np
import matplotlib.pyplot as plt

def plot_and_save_npz(npz_path="koopman_test_dataset.npz", save_base_dir="test_dataset_plots"):
    """
    读取 npz 文件，遍历各个数据段的变量并作图保存。
    """
    # 1. 检查文件是否存在并创建主保存目录
    if not os.path.exists(npz_path):
        print(f"错误: 找不到文件 {npz_path}，请确认路径是否正确。")
        return

    os.makedirs(save_base_dir, exist_ok=True)
    
    # 2. 加载数据
    print(f">>> 开始加载数据: {npz_path}")
    try:
        data = np.load(npz_path, allow_pickle=True)['datas']
    except Exception as e:
        print(f"读取 npz 文件失败: {e}")
        return

    num_segments = len(data)
    print(f">>> 共发现 {num_segments} 个数据段 (Segments)。开始绘图...\n")

    # 3. 遍历每一个数据段
    for i, seg in enumerate(data):
        time_len = seg.get('len', 0)
        if time_len == 0:
            continue
            
        time_steps = np.arange(time_len)
        
        # 为当前数据段创建一个专属的子文件夹
        seg_dir = os.path.join(save_base_dir, f"segment_{i:03d}")
        os.makedirs(seg_dir, exist_ok=True)
        
        print(f"正在处理第 {i+1}/{num_segments} 段数据 (长度: {time_len}) -> 保存至 {seg_dir}")
        
        # 4. 遍历该段中的所有物理变量
        for key, value in seg.items():
            if key == 'len': # 跳过长度标记
                continue
                
            if not isinstance(value, np.ndarray):
                continue
                
            # ----------------------------------------
            # 情况 A: 1维数组 (shape: [time])
            # ----------------------------------------
            if value.ndim == 1:
                plt.figure(figsize=(10, 4))
                plt.plot(time_steps, value, color='b', linewidth=1.5)
                plt.title(f'Segment {i} - {key}', fontsize=14)
                plt.xlabel('Time Step', fontsize=12)
                plt.ylabel('Value', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                save_path = os.path.join(seg_dir, f"{key}.png")
                plt.savefig(save_path, dpi=150)
                plt.close() # 必须 close 释放内存
                
            # ----------------------------------------
            # 情况 B: 2维数组 (shape: [dim, time])
            # ----------------------------------------
            elif value.ndim == 2:
                dim, t = value.shape
                
                # 容错检查：如果时间维度在前面，转置一下（根据你之前的代码，通常时间在最后一维）
                if t != time_len and dim == time_len:
                    value = value.T
                    dim, t = value.shape
                    
                # 动态调整图片高度，维度越多图片越高
                fig_height = max(3, 2 * dim) 
                fig, axes = plt.subplots(dim, 1, figsize=(12, fig_height), sharex=True)
                
                # 如果只有1个维度，axes 不是列表，为了统一遍历将其包装为列表
                if dim == 1:
                    axes = [axes]
                    
                fig.suptitle(f'Segment {i} - {key}', fontsize=16)
                
                for d in range(dim):
                    axes[d].plot(time_steps, value[d, :], linewidth=1.5, label=f'Dim {d}')
                    axes[d].set_ylabel(f'Dim {d}', fontsize=10)
                    axes[d].grid(True, linestyle='--', alpha=0.7)
                    axes[d].legend(loc='upper right')
                    
                axes[-1].set_xlabel('Time Step', fontsize=12)
                plt.tight_layout()
                
                save_path = os.path.join(seg_dir, f"{key}.png")
                plt.savefig(save_path, dpi=150)
                plt.close()

    print(f"\n>>> 绘图完成！所有图像已保存在: ./{save_base_dir}/ 目录下。")

if __name__ == "__main__":
    plot_and_save_npz()