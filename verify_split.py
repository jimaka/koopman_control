import os
import numpy as np
import matplotlib.pyplot as plt

def verify_segment_split_aligned(npy_path="koopman_dataset_v1.npz", ratios=(0.7, 0.15, 0.15)):
    """
    段内切分验证脚本 (严格对齐训练脚本的切片逻辑)
    """
    # 1. 创建保存图片的专属文件夹
    save_dir = "verify_plots"
    os.makedirs(save_dir, exist_ok=True)
    print(f"验证图片将保存在目录: ./{save_dir}/\n")

    print(f"正在加载数据文件: {npy_path} ...")
    try:
        data_raw = np.load(npy_path, allow_pickle=True)
        segments = data_raw['datas']
    except Exception as e:
        print(f"读取失败，请检查文件是否存在: {e}")
        return

    train_r, val_r, test_r = ratios
    print(f"切分比例: Train={train_r*100}%, Val={val_r*100}%, Test={test_r*100}%")
    print(f"成功加载，共包含 {len(segments)} 个航行段。\n")
    print("="*60)

    for i, seg in enumerate(segments):
        L = seg['len']
        
        # ====================================================
        # 【严格对齐训练脚本】的索引计算逻辑
        # 参考 IntraSegmentKoopmanDataset 的切分方式
        # ====================================================
        n_train = int(L * train_r)
        n_val = int(L * val_r)
        
        idx_train_start, idx_train_end = 0, n_train
        idx_val_start,   idx_val_end   = n_train, n_train + n_val
        idx_test_start,  idx_test_end  = n_train + n_val, L
        
        # 提取全局位置数据，原始形状为 (2, L)
        pos_all = seg['Pos']  
        
        # 按时间序贯切片
        pos_train = pos_all[:, idx_train_start:idx_train_end]
        pos_val   = pos_all[:, idx_val_start:idx_val_end]
        pos_test  = pos_all[:, idx_test_start:idx_test_end]
        
        # 打印验证信息
        print(f"航行段 [{i}] (总长度: {L}) 切分明细:")
        print(f"  - Train: 索引 [{idx_train_start} : {idx_train_end}] | 长度 {pos_train.shape[1]}")
        print(f"  - Val  : 索引 [{idx_val_start} : {idx_val_end}] | 长度 {pos_val.shape[1]}")
        print(f"  - Test : 索引 [{idx_test_start} : {idx_test_end}] | 长度 {pos_test.shape[1]}")
        
        # ====================================================
        # 可视化三段轨迹的物理接力关系
        # ====================================================
        plt.figure(figsize=(10, 8))
        
        # 绘制全集底色 (浅灰色虚线)
        plt.plot(pos_all[0, :], pos_all[1, :], color='gray', linestyle=':', 
                 linewidth=1, alpha=0.4, label='Full Trajectory')
        
        # 绘制 Train (蓝色)
        if pos_train.shape[1] > 0:
            plt.plot(pos_train[0, :], pos_train[1, :], color='#1f77b4', 
                     linewidth=3.0, label=f'Train (First {train_r*100}%)')
            # 标记运动起点
            plt.scatter(pos_train[0, 0], pos_train[1, 0], c='green', s=80, 
                        marker='o', zorder=5, label='Start Point')
        
        # 绘制 Val (橙色)
        if pos_val.shape[1] > 0:
            plt.plot(pos_val[0, :], pos_val[1, :], color='#ff7f0e', 
                     linewidth=3.0, label=f'Val (Middle {val_r*100}%)')
        
        # 绘制 Test (红色)
        if pos_test.shape[1] > 0:
            plt.plot(pos_test[0, :], pos_test[1, :], color='#d62728', 
                     linewidth=3.0, label=f'Test (Last {test_r*100}%)')
            # 标记运动终点
            plt.scatter(pos_test[0, -1], pos_test[1, -1], c='purple', s=80, 
                        marker='X', zorder=5, label='End Point')

        plt.title(f"Segment {i} Intra-Sequential Split Verification", fontsize=14)
        plt.xlabel("Global X [m]", fontsize=12)
        plt.ylabel("Global Y [m]", fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal') # 保证 X 和 Y 轴比例一致，展示真实的转弯半径
        
        # 拼接保存路径
        save_path = os.path.join(save_dir, f"segment_{i}_split.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  -> 可视化图片已保存至: {save_path}")
        print("-" * 60)

if __name__ == "__main__":
    # 注意：这里的名字替换为你实际的数据集文件名
    verify_segment_split_aligned("koopman_dataset_v1.npz", ratios=(0.7, 0.15, 0.15))