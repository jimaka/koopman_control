import numpy as np
import matplotlib
# 强制使用 Agg 后端，确保在 Docker/无显示器环境下运行不报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os

def visualize_and_save_all(npz_path, output_folder="analysis_plots"):
    """
    读取 npz 文件，为每个数据段生成图表并保存到文件夹中
    """
    # 1. 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建文件夹: {output_folder}")

    # 2. 加载数据
    if not os.path.exists(npz_path):
        print(f"错误: 找不到文件 {npz_path}")
        return

    data = np.load(npz_path, allow_pickle=True)
    datas = data['datas']
    print(f"成功加载 {len(datas)} 个数据段。开始绘图...")

    # 对应 C++ 逻辑的阶段名称（可选）
    phase_tags = [
        "01_Forward_Step", "02_Reverse_Step", "03_Single_L", 
        "04_Single_R", "05_ZigZag", "06_Chirp", "07_PRBS"
    ]

    for idx, seg in enumerate(datas):
        # 确定文件名
        tag = phase_tags[idx] if idx < len(phase_tags) else f"Segment_{idx}"
        save_path = os.path.join(output_folder, f"{tag}.png")

        # 提取数据 (D, N) 格式
        pos = seg['Pos']
        vel = seg['Vel']
        pqr = seg['pqr']
        euler = seg['Euler']
        cmd = seg['Thrusters_CMD']
        steps = np.arange(seg['len'])

        # 创建画布
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Experimental Analysis: {tag}", fontsize=20)

        # 图 1: 2D 轨迹 (X-Y)
        axes[0,0].plot(pos[0, :], pos[1, :], 'b-')
        axes[0,0].set_title("Trajectory (X-Y)")
        axes[0,0].set_xlabel("X (m)")
        axes[0,0].set_ylabel("Y (m)")
        axes[0,0].axis('equal')
        axes[0,0].grid(True)

        # 图 2: 线速度 (Surge/Sway)
        axes[0,1].plot(steps, vel[0, :], label='u (surge)')
        axes[0,1].plot(steps, vel[1, :], label='v (sway)')
        axes[0,1].set_title("Linear Velocity")
        axes[0,1].legend()
        axes[0,1].grid(True)

        # 图 3: 偏航角 (Yaw)
        axes[0,2].plot(steps, np.rad2deg(euler[2, :]), 'r-')
        axes[0,2].set_title("Yaw Angle (deg)")
        axes[0,2].set_ylabel("Degrees")
        axes[0,2].grid(True)

        # 图 4: 油门指令 (Throttles)
        axes[1,0].plot(steps, cmd[0, :], label='Port')
        axes[1,0].plot(steps, cmd[2, :], label='Starboard')
        axes[1,0].set_title("Thruster Throttles (%)")
        axes[1,0].legend()
        axes[1,0].grid(True)

        # 图 5: 舵角指令 (Angles)
        axes[1,1].plot(steps, cmd[1, :], label='Port')
        axes[1,1].plot(steps, cmd[3, :], label='Starboard')
        axes[1,1].set_title("Thruster Angles (deg)")
        axes[1,1].legend()
        axes[1,1].grid(True)

        # 图 6: 角速度 (Yaw Rate)
        axes[1,2].plot(steps, pqr[0, :], 'g-')
        axes[1,2].set_title("Yaw Rate (r)")
        axes[1,2].grid(True)

        # 调整布局并保存
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path)
        plt.close(fig) # 释放内存
        print(f"[{idx+1}/{len(datas)}] 已保存: {save_path}")

    print("\n所有图表已生成完毕。")

if __name__ == "__main__":
    # 请确保该文件名与你生成的 npz 文件名一致
    visualize_and_save_all("koopman_dataset_v1.npz")