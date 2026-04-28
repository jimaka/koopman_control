import rosbag
import numpy as np
from scipy.interpolate import interp1d
import os

def align_and_resample(raw_data, target_hz=0.5):
    """
    针对船舶水平面运动(3-DOF)对齐数据。
    """
    if len(raw_data['odom_ts']) < 2 or len(raw_data['cmd_ts']) < 2:
        print(f"数据不足: odom_ts={len(raw_data['odom_ts'])}, cmd_ts={len(raw_data['cmd_ts'])}")
        return None

    ts_odom = np.array(raw_data['odom_ts'])
    ts_cmd = np.array(raw_data['cmd_ts'])
    
    t_start = max(ts_odom[0], ts_cmd[0])
    t_end = min(ts_odom[-1], ts_cmd[-1])
    
    print(f"时间范围: odom=[{ts_odom[0]:.2f}, {ts_odom[-1]:.2f}], cmd=[{ts_cmd[0]:.2f}, {ts_cmd[-1]:.2f}]")
    print(f"重叠范围: [{t_start:.2f}, {t_end:.2f}]")
    
    if t_start >= t_end:
        print(f"错误: 数据时间不重叠: t_start={t_start:.2f}, t_end={t_end:.2f}")
        return None

    dt = 1.0 / target_hz
    t_common = np.arange(t_start, t_end, dt)
    
    print(f"重采样时间点: {len(t_common)} 个点, dt={dt:.3f}s")
    
    aligned = {'time': t_common}
    
    # --- 水平面状态处理 ---
    # 1. 位置: 只取 X 和 Y
    print(f"位置数据形状: {raw_data['Pos'].shape}")
    f_pos = interp1d(ts_odom, raw_data['Pos'], axis=0, kind='linear', fill_value="extrapolate")
    aligned['Pos'] = f_pos(t_common).astype(np.float32) # (N, 2)
    print(f"对齐后位置形状: {aligned['Pos'].shape}")
    
    # 2. 速度: 只取线速度 X 和 Y
    print(f"速度数据形状: {raw_data['Vel'].shape}")
    f_vel = interp1d(ts_odom, raw_data['Vel'], axis=0, kind='linear', fill_value="extrapolate")
    aligned['Vel'] = f_vel(t_common).astype(np.float32) # (N, 2)
    print(f"对齐后速度形状: {aligned['Vel'].shape}")

    # 3. 角速度: 只取偏航速率 r (z轴分量)
    print(f"角速度数据形状: {raw_data['pqr'].shape}")
    f_pqr = interp1d(ts_odom, raw_data['pqr'], axis=0, kind='linear', fill_value="extrapolate")
    aligned['pqr'] = f_pqr(t_common).astype(np.float32) # (N, 1)
    print(f"对齐后角速度形状: {aligned['pqr'].shape}")

    # 4. 偏航角 (Yaw): 处理跳变
    print(f"偏航角数据形状: {raw_data['Yaw'].shape}")
    yaw_raw = np.array(raw_data['Yaw'])
    yaw_unwrapped = np.unwrap(yaw_raw) 
    f_yaw = interp1d(ts_odom, yaw_unwrapped, kind='linear', fill_value="extrapolate")
    aligned['Euler'] = f_yaw(t_common).reshape(-1, 1).astype(np.float32) # (N, 1)
    print(f"对齐后偏航角形状: {aligned['Euler'].shape}")
    
    # 5. 控制指令 (Thrusters_CMD)
    print(f"控制指令数据形状: {raw_data['Thrusters_CMD'].shape}")
    f_cmd = interp1d(ts_cmd, raw_data['Thrusters_CMD'], axis=0, kind='linear', fill_value="extrapolate")
    aligned['Thrusters_CMD'] = f_cmd(t_common).astype(np.float32)
    print(f"对齐后控制指令形状: {aligned['Thrusters_CMD'].shape}")
    
    aligned['len'] = len(t_common)
    
    # 打印数据统计
    print(f"\n数据统计:")
    print(f"  时间长度: {t_common[-1] - t_common[0]:.2f}s")
    print(f"  采样频率: {target_hz}Hz")
    print(f"  总数据点: {len(t_common)}")
    
    return aligned

def process_bag(bag_path, target_hz=0.5):
    print(f"\n处理文件: {bag_path}")
    
    raw_data = {
        'odom_ts': [], 'Pos': [], 'Vel': [], 'Yaw': [], 'pqr': [],
        'cmd_ts': [], 'Thrusters_CMD': []
    }
    
    # 检查文件是否存在
    if not os.path.exists(bag_path):
        print(f"错误: 文件不存在 {bag_path}")
        return None
    
    try:
        bag = rosbag.Bag(bag_path)
        TOPIC_POSE = '/localization/fusion_pose'
        TOPIC_THRUSTER = '/system/chassis_feedback'
        # TOPIC_THRUSTER = '/state/controller_state'  # 替换为实际的控制指令话题
        
        print(f"读取话题: {TOPIC_POSE}, {TOPIC_THRUSTER}")
        
        # 统计消息数量
        odom_count = 0
        thruster_count = 0
        
        # 修复: bag.read_messages() 返回三元组 (topic, msg, t)
        for topic, msg, t in bag.read_messages(topics=[TOPIC_POSE, TOPIC_THRUSTER]):
            t_sec = t.to_sec()
            
            if topic == TOPIC_POSE:
                odom_count += 1
                raw_data['odom_ts'].append(t_sec)
                # 水平面只需要 X, Y
                raw_data['Pos'].append([msg.position.x, msg.position.y])
                # 线速度 X, Y
                raw_data['Vel'].append([msg.velocity.x, msg.velocity.y])
                # 角速度只取 Z (Yaw Rate)
                raw_data['pqr'].append([msg.angular_velocity.z]) 
                # 姿态只取 Yaw
                raw_data['Yaw'].append(msg.rpy.yaw)

            elif topic == TOPIC_THRUSTER:
                thruster_count += 1
                raw_data['cmd_ts'].append(t_sec)
                # 保持 4 维控制向量
                u = [
                    msg.port_thruster_throttle,
                    msg.port_thruster_angle,
                    msg.starboard_thruster_throttle,
                    msg.starboard_thruster_angle
                ]
                raw_data['Thrusters_CMD'].append(u)
        
        bag.close()
        
        print(f"读取完成: odom消息={odom_count}, thruster消息={thruster_count}")
        
        # 转换为numpy数组
        for k in raw_data:
            if len(raw_data[k]) > 0:
                raw_data[k] = np.array(raw_data[k])
                print(f"  {k}: shape={raw_data[k].shape}, dtype={raw_data[k].dtype if hasattr(raw_data[k], 'dtype') else 'N/A'}")
            else:
                print(f"  {k}: 无数据")
                raw_data[k] = np.array([])
        
        return raw_data
        
    except Exception as e:
        print(f"处理bag文件时出错: {e}")
        return None

def main():
    bag_files = ["../replay.bag"]  # 替换为你的文件名
    output_npz = "sim_10HZ.npz"  # 使用不同的文件名避免混淆
    
    all_datas = []
    
    print("开始处理ROS bag文件...")
    
    for i, bf in enumerate(bag_files):
        print(f"\n=== 处理第 {i+1} 个文件 ===")
        
        # 处理原始数据
        raw_data = process_bag(bf, target_hz=10)
        
        if raw_data is None:
            print(f"无法处理文件: {bf}")
            continue
            
        # 对齐和重采样
        print("\n进行数据对齐和重采样...")
        aligned_data = align_and_resample(raw_data, target_hz=10)
        
        if aligned_data:
            # 将数据转换为Koopman训练脚本期望的格式
            data_dict = {}
            for key, value in aligned_data.items():
                if key == 'time':
                    continue  # 时间戳不保存到data_dict
                elif key == 'len':
                    data_dict[key] = int(value)
                elif key == 'Euler':
                    # 为了与原始格式兼容，保持3维欧拉角
                    # 注意: 这里我们只有yaw，但可能需要补充roll和pitch为0
                    euler_full = np.zeros((len(value), 3), dtype=np.float32)
                    euler_full[:, 2] = value.flatten()  # yaw在第三维
                    data_dict[key] = euler_full.T  # 转置为 (3, N)
                    print(f"欧拉角最终形状: {data_dict[key].shape}")
                else:
                    # 其他数据保持 (N, D) 形状
                    data_dict[key] = value.T  # 转置为 (D, N)
                    print(f"{key}最终形状: {data_dict[key].shape}")
            
            all_datas.append(data_dict)
            print(f"文件 {bf} 处理完成，添加到飞行数据列表")
    
    # 保存数据
    if all_datas:
        # 存储为 Koopman 训练脚本识别的格式
        np.savez_compressed(output_npz, datas=np.array(all_datas, dtype=object))
        print(f"\n处理完成，保存了 {len(all_datas)} 段飞行数据到 {output_npz}")
        
        # 打印保存的数据信息
        print("\n保存的数据结构:")
        for i, data in enumerate(all_datas):
            print(f"\n飞行 {i+1}:")
            for key in sorted(data.keys()):
                if key == 'len':
                    print(f"  {key}: {data[key]}")
                else:
                    print(f"  {key}: shape={data[key].shape}")
    else:
        print("\n错误: 没有成功处理任何文件")

if __name__ == "__main__":
    main()