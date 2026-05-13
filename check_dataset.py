import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_single_segment(segments, segment_idx, save_dir='dataset_plots'):
    """
    核心绘图函数：绘制指定航段的所有变量
    """
    if segment_idx < 0 or segment_idx >= len(segments):
        print(f"❌ 索引越界！数据集仅有 {len(segments)} 段数据 (索引范围 0 ~ {len(segments)-1})。")
        return False

    seg = segments[segment_idx]
    T = seg['len']
    time_steps = np.arange(T)

    # ==========================================
    # 动态提取所有维度的变量
    # ==========================================
    pos_names   = ['Pos X (Surge/North)', 'Pos Y (Sway/East)', 'Pos Z (Heave/Down)']
    euler_names = ['Euler Roll (phi)', 'Euler Pitch (theta)', 'Euler Yaw (psi)']
    vel_names   = ['Vel u (Surge)', 'Vel v (Sway)', 'Vel w (Heave)']
    pqr_names   = ['Rate p (Roll rate)', 'Rate q (Pitch rate)', 'Rate r (Yaw rate)']
    
    variables = []
    
    for i in range(seg['Pos'].shape[0]):
        variables.append((pos_names[i] if i < 3 else f'Pos {i}', seg['Pos'][i, :T], 'royalblue'))
    for i in range(seg['Euler'].shape[0]):
        variables.append((euler_names[i] if i < 3 else f'Euler {i}', seg['Euler'][i, :T], 'forestgreen'))
    for i in range(seg['Vel'].shape[0]):
        variables.append((vel_names[i] if i < 3 else f'Vel {i}', seg['Vel'][i, :T], 'darkorange'))
    for i in range(seg['pqr'].shape[0]):
        variables.append((pqr_names[i] if i < 3 else f'pqr {i}', seg['pqr'][i, :T], 'purple'))
    for i in range(seg['Thrusters_CMD'].shape[0]):
        variables.append((f'Thruster {i+1} CMD', seg['Thrusters_CMD'][i, :T], 'crimson'))

    total_vars = len(variables)

    # ==========================================
    # 自动计算网格并绘图
    # ==========================================
    cols = 4
    rows = int(np.ceil(total_vars / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3))
    fig.suptitle(f"Comprehensive 6-DOF States & Controls (Segment {segment_idx})", fontsize=20, y=1.02)
    
    axes = axes.flatten()

    for idx, (name, data_array, color) in enumerate(variables):
        ax = axes[idx]
        if 'Thruster' in name:
            ax.step(time_steps, data_array, color=color, linewidth=1.5, where='post')
        else:
            ax.plot(time_steps, data_array, color=color, linewidth=1.5)
            
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if idx >= total_vars - cols:
            ax.set_xlabel("Time Step")
            
    for i in range(total_vars, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"all_variables_segment_{segment_idx:03d}.png")
    
    # 捕获可能的绘图警告
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        
    plt.close(fig) # 极其重要：释放内存，防止批量绘图时内存溢出
    return True

def main():
    parser = argparse.ArgumentParser(description="船舶多段航行数据全变量绘图工具")
    parser.add_argument('--data', type=str, default='koopman_train_merged.npz', help='NPZ 数据集路径')
    parser.add_argument('--seg', type=int, default=-1, 
                        help='要绘制的数据段索引。输入具体数字(如 0, 5)画单段；输入 -1 表示画出所有段 (默认: -1)')
    parser.add_argument('--out', type=str, default='dataset_plots', help='图片保存的文件夹名')
    args = parser.parse_args()

    print(f"🔍 正在一次性加载数据集: {args.data} ...")
    try:
        data = np.load(args.data, allow_pickle=True)
        segments = data['datas']
    except Exception as e:
        print(f"❌ 加载失败，请检查文件路径: {e}")
        return

    num_segments = len(segments)
    os.makedirs(args.out, exist_ok=True)
    print(f"✅ 数据集加载成功，共包含 {num_segments} 个连续航段。")

    if args.seg == -1:
        # 批量绘制所有航段
        print("🚀 开始批量绘图模式...")
        success_count = 0
        for i in range(num_segments):
            print(f"  -> 正在绘制 [{i+1:03d}/{num_segments:03d}] 段...")
            if plot_single_segment(segments, i, args.out):
                success_count += 1
        print("=" * 50)
        print(f"🎉 批量绘图完成！共成功保存 {success_count} 张高清全景图至目录: '{args.out}/'")
        
    else:
        # 绘制单一特定航段
        print(f"🎨 开始单次绘图模式 (Segment {args.seg})...")
        if plot_single_segment(segments, args.seg, args.out):
            print(f"✅ 绘制成功！图片已保存至目录: '{args.out}/'")

if __name__ == '__main__':
    main()