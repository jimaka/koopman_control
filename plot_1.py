import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_npz_data(npz_file):
    """简洁的NPZ数据可视化"""
    
    # 加载数据
    data = np.load(npz_file, allow_pickle=True)
    
    # 获取所有键
    keys = list(data.keys())
    print(f"NPZ文件包含以下键: {keys}")
    
    for key in keys:
        value = data[key]
        print(f"\n{key}: {type(value)}, ", end="")
        
        if isinstance(value, np.ndarray):
            print(f"shape={value.shape}")
            
            # 可视化数组
            if value.ndim == 1:
                plt.figure(figsize=(12, 4))
                plt.plot(value)
                plt.title(f"{key} (1D array, shape={value.shape})")
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{key}_1d.png", dpi=150, bbox_inches='tight')
                plt.close()
                
            elif value.ndim == 2:
                rows, cols = value.shape
                plt.figure(figsize=(14, 5))
                
                # 热图
                plt.subplot(1, 2, 1)
                plt.imshow(value, aspect='auto', cmap='viridis')
                plt.colorbar()
                plt.title(f"{key} (2D, {rows}×{cols})")
                plt.xlabel('Column')
                plt.ylabel('Row')
                
                # 统计
                plt.subplot(1, 2, 2)
                plt.axis('off')
                stats_text = f"""
                Shape: {value.shape}
                Min: {value.min():.4f}
                Max: {value.max():.4f}
                Mean: {value.mean():.4f}
                Std: {value.std():.4f}
                """
                plt.text(0.1, 0.5, stats_text, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.tight_layout()
                plt.savefig(f"{key}_2d.png", dpi=150, bbox_inches='tight')
                plt.close()
                
            elif value.ndim == 3:
                # 显示前3个切片
                depth = value.shape[0]
                slices = min(3, depth)
                
                fig, axes = plt.subplots(1, slices, figsize=(5*slices, 4))
                if slices == 1:
                    axes = [axes]
                
                for i in range(slices):
                    axes[i].imshow(value[i], aspect='auto', cmap='viridis')
                    axes[i].set_title(f"Slice {i}/{depth-1}")
                    axes[i].set_xlabel('Column')
                    axes[i].set_ylabel('Row')
                
                plt.suptitle(f"{key} (3D, shape={value.shape})")
                plt.tight_layout()
                plt.savefig(f"{key}_3d.png", dpi=150, bbox_inches='tight')
                plt.close()
                
            else:
                # 高维数组
                plt.figure(figsize=(10, 6))
                plt.axis('off')
                
                stats_text = f"""
                {key} (高维数组)
                Shape: {value.shape}
                Total elements: {value.size}
                Min: {value.min():.6f}
                Max: {value.max():.6f}
                Mean: {value.mean():.6f}
                Std: {value.std():.6f}
                """
                plt.text(0.1, 0.5, stats_text, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
                
                plt.tight_layout()
                plt.savefig(f"{key}_high_dim.png", dpi=150, bbox_inches='tight')
                plt.close()
        
        elif isinstance(value, dict):
            print(f"dict with keys: {list(value.keys())}")
            # 保存字典内容
            import json
            with open(f"{key}_dict.json", 'w') as f:
                json_dict = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_dict[k] = v.tolist() if v.size < 1000 else f"数组 shape={v.shape}"
                    else:
                        json_dict[k] = str(v)
                json.dump(json_dict, f, indent=2, default=str)
        
        else:
            print(f"value: {value}")
    
    data.close()
    print(f"\n所有可视化图表已保存为PNG文件")

def visualize_dict_in_npz(npz_file, dict_key='datas'):
    """专门处理包含字典的NPZ文件"""
    
    # 加载数据
    data = np.load(npz_file, allow_pickle=True)
    
    # 提取字典
    dict_array = data[dict_key]
    if dict_array.size == 1:
        flight_data = dict_array.item()
    else:
        flight_data = dict_array.flat[0]
    
    print(f"字典键: {list(flight_data.keys())}")
    
    # 创建综合图表
    fig = plt.figure(figsize=(20, 15))
    
    # 检查数据长度
    n_samples = flight_data.get('len', 0)
    if n_samples == 0:
        # 尝试从数组形状推断
        for key, value in flight_data.items():
            if isinstance(value, np.ndarray):
                n_samples = value.shape[1] if value.ndim == 2 else value.shape[0]
                break
    
    time = np.arange(n_samples) * 0.1  # 假设10Hz采样
    
    # 1. 位置数据
    if 'Pos' in flight_data:
        pos = flight_data['Pos']
        if pos.shape[0] == 2:  # (2, N)
            ax1 = plt.subplot(4, 3, 1)
            ax1.plot(pos[0], pos[1])
            ax1.set_title('XY Trajectory')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            
            ax2 = plt.subplot(4, 3, 2)
            ax2.plot(time, pos[0])
            ax2.set_title('X Position')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('X (m)')
            ax2.grid(True, alpha=0.3)
            
            ax3 = plt.subplot(4, 3, 3)
            ax3.plot(time, pos[1])
            ax3.set_title('Y Position')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Y (m)')
            ax3.grid(True, alpha=0.3)
    
    # 2. 速度数据
    if 'Vel' in flight_data:
        vel = flight_data['Vel']
        if vel.shape[0] == 2:  # (2, N)
            ax4 = plt.subplot(4, 3, 4)
            ax4.plot(time, vel[0])
            ax4.set_title('X Velocity')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Vx (m/s)')
            ax4.grid(True, alpha=0.3)
            
            ax5 = plt.subplot(4, 3, 5)
            ax5.plot(time, vel[1])
            ax5.set_title('Y Velocity')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Vy (m/s)')
            ax5.grid(True, alpha=0.3)
            
            ax6 = plt.subplot(4, 3, 6)
            speed = np.sqrt(vel[0]**2 + vel[1]**2)
            ax6.plot(time, speed)
            ax6.set_title('Speed')
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Speed (m/s)')
            ax6.grid(True, alpha=0.3)
    
    # 3. 偏航角数据
    if 'Euler' in flight_data:
        euler = flight_data['Euler']
        if euler.shape[0] == 3:  # (3, N)
            ax7 = plt.subplot(4, 3, 7)
            yaw_deg = euler[2]
            ax7.plot(time, yaw_deg)
            ax7.set_title('Yaw Angle')
            ax7.set_xlabel('Time (s)')
            ax7.set_ylabel('Yaw (deg)')
            ax7.grid(True, alpha=0.3)
    
    # 4. 角速度数据
    if 'pqr' in flight_data:
        pqr = flight_data['pqr']
        ax8 = plt.subplot(4, 3, 8)
        if pqr.shape[0] == 3:  # (3, N)
            yaw_rate = np.degrees(pqr[2])
        else:  # (1, N)
            yaw_rate = np.degrees(pqr[0])
        ax8.plot(time, yaw_rate)
        ax8.set_title('Yaw Rate')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Yaw Rate (deg/s)')
        ax8.grid(True, alpha=0.3)
    
    # 5. 控制指令数据
    if 'Thrusters_CMD' in flight_data:
        cmd = flight_data['Thrusters_CMD']
        if cmd.shape[0] == 4:  # (4, N)
            ax9 = plt.subplot(4, 3, 9)
            ax9.plot(time, cmd[0], label='Port Throttle')
            ax9.plot(time, cmd[2], label='Starboard Throttle')
            ax9.set_title('Throttle Commands')
            ax9.set_xlabel('Time (s)')
            ax9.set_ylabel('Throttle')
            ax9.grid(True, alpha=0.3)
            ax9.legend(fontsize=8)
            
            ax10 = plt.subplot(4, 3, 10)
            ax10.plot(time, cmd[1], label='Port Angle')
            ax10.plot(time, cmd[3], label='Starboard Angle')
            ax10.set_title('Angle Commands')
            ax10.set_xlabel('Time (s)')
            ax10.set_ylabel('Angle (rad)')
            ax10.grid(True, alpha=0.3)
            ax10.legend(fontsize=8)
    
    # 6. 统计信息
    ax11 = plt.subplot(4, 3, (11, 12))
    ax11.axis('off')
    
    stats_text = "Data Statistics\n"
    stats_text += "=" * 40 + "\n\n"
    
    for key, value in flight_data.items():
        if key == 'len':
            stats_text += f"data_length: {value}\n"
        elif isinstance(value, np.ndarray):
            stats_text += f"{key}: shape={value.shape}"
            if value.ndim <= 2:
                stats_text += f", range=[{value.min():.3f}, {value.max():.3f}]"
            stats_text += "\n"
    
    ax11.text(0.05, 0.95, stats_text, fontsize=10,
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.suptitle(f'Flight Data Visualization - {npz_file}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data_comprehensive.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    data.close()
    print("综合图表已保存为 flight_data_comprehensive.png")

def auto_visualize(npz_file):
    """自动检测并可视化NPZ文件"""
    
    print(f"处理文件: {npz_file}")
    
    # 先探索数据结构
    data = np.load(npz_file, allow_pickle=True)
    keys = list(data.keys())
    
    print(f"文件包含以下键: {keys}")
    
    # 检查是否是包含字典的对象数组
    for key in keys:
        value = data[key]
        if isinstance(value, np.ndarray) and value.dtype == object:
            print(f"检测到对象数组: {key}, shape={value.shape}")
            if value.size > 0 and isinstance(value.flat[0], dict):
                print("包含字典数据，使用专门的可视化方法")
                data.close()
                visualize_dict_in_npz(npz_file, key)
                return
    
    data.close()
    
    # 否则使用通用可视化
    visualize_npz_data(npz_file)

if __name__ == "__main__":
    # 查找当前目录下的npz文件
    npz_files = [f for f in os.listdir('.') if f.endswith('.npz')]
    
    if not npz_files:
        print("当前目录下未找到.npz文件")
    else:
        for npz_file in npz_files:
            print(f"\n{'='*60}")
            print(f"开始处理: {npz_file}")
            print('='*60)
            
            try:
                auto_visualize(npz_file)
                print(f"完成处理: {npz_file}")
            except Exception as e:
                print(f"处理 {npz_file} 时出错: {e}")