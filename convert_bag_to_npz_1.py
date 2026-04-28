import rosbag
import numpy as np
from scipy.interpolate import interp1d
import os

# =================================================================
# 配置区：根据 KoopmanGranularPilot.hpp 定义的实验阶段
# =================================================================
CPP_PHASES = [
    # (开始时间s, 结束时间s, 阶段标签)
    (0.0,     1499.0,  "STR_STEP_FORWARD"),   # 直线加速
    (1501.0,  2999.0,  "STR_STEP_BACK"),      # 倒车
    (3001.0,  4999.0,  "SINGLE_THRUSTER_L"),  # 左单推
    (5001.0,  6999.0,  "SINGLE_THRUSTER_R"),  # 右单推
    (7001.0,  12999.0, "ZIGZAG_ALL"),         # Z型试验组
    (13001.0, 13999.0, "CHIRP_EXCITATION"),   # Chirp信号
    (14001.0, 14999.0, "PRBS_EXCITATION"),    # PRBS信号
]

# 话题定义
TOPIC_POSE = '/localization/fusion_pose'
TOPIC_THRUSTER = '/system/chassis_feedback'

def process_bag(bag_path):
    """从 Bag 中提取原始离散数据"""
    raw_data = {
        'odom_ts': [], 'Pos': [], 'Vel': [], 'Yaw': [], 'pqr': [],
        'cmd_ts': [], 'Thrusters_CMD': []
    }
    
    if not os.path.exists(bag_path):
        print(f"错误: 文件不存在 {bag_path}")
        return None
    
    print(f"正在读取 Bag: {bag_path}")
    try:
        bag = rosbag.Bag(bag_path)
        for topic, msg, t in bag.read_messages(topics=[TOPIC_POSE, TOPIC_THRUSTER]):
            t_sec = t.to_sec()
            
            if topic == TOPIC_POSE:
                raw_data['odom_ts'].append(t_sec)
                raw_data['Pos'].append([msg.position.x, msg.position.y])
                raw_data['Vel'].append([msg.velocity.x, msg.velocity.y])
                raw_data['pqr'].append([msg.angular_velocity.z]) 
                raw_data['Yaw'].append(msg.rpy.yaw)

            elif topic == TOPIC_THRUSTER:
                raw_data['cmd_ts'].append(t_sec)
                # 对应 4 维控制向量 [左油门, 左舵角, 右油门, 右舵角]
                u = [
                    msg.port_thruster_throttle, msg.port_thruster_angle,
                    msg.starboard_thruster_throttle, msg.starboard_thruster_angle
                ]
                raw_data['Thrusters_CMD'].append(u)
        bag.close()
        return raw_data
    except Exception as e:
        print(f"读取失败: {e}")
        return None

def align_and_resample(raw_data, target_hz=10.0):
    """对齐数据并将时间轴归一化为从 0 开始"""
    ts_odom = np.array(raw_data['odom_ts'])
    ts_cmd = np.array(raw_data['cmd_ts'])
    
    if len(ts_odom) < 2 or len(ts_cmd) < 2:
        return None

    # 时间轴同步：以第一个到达的消息为 T=0
    t0 = min(ts_odom[0], ts_cmd[0])
    ts_odom -= t0
    ts_cmd -= t0
    
    # 选取交集范围
    t_start = max(ts_odom[0], ts_cmd[0])
    t_end = min(ts_odom[-1], ts_cmd[-1])
    
    dt = 1.0 / target_hz
    t_common = np.arange(t_start, t_end, dt)
    
    aligned = {'time': t_common}
    
    # 线性插值处理
    f_pos = interp1d(ts_odom, raw_data['Pos'], axis=0, kind='linear', fill_value="extrapolate")
    aligned['Pos'] = f_pos(t_common).astype(np.float32)
    
    f_vel = interp1d(ts_odom, raw_data['Vel'], axis=0, kind='linear', fill_value="extrapolate")
    aligned['Vel'] = f_vel(t_common).astype(np.float32)

    f_pqr = interp1d(ts_odom, raw_data['pqr'], axis=0, kind='linear', fill_value="extrapolate")
    aligned['pqr'] = f_pqr(t_common).astype(np.float32)

    # 偏航角解算
    yaw_unwrapped = np.unwrap(raw_data['Yaw']) 
    f_yaw = interp1d(ts_odom, yaw_unwrapped, kind='linear', fill_value="extrapolate")
    aligned['Euler'] = f_yaw(t_common).reshape(-1, 1).astype(np.float32)
    
    f_cmd = interp1d(ts_cmd, raw_data['Thrusters_CMD'], axis=0, kind='linear', fill_value="extrapolate")
    aligned['Thrusters_CMD'] = f_cmd(t_common).astype(np.float32)
    
    aligned['len'] = len(t_common)
    return aligned

def split_by_logic(aligned_data, target_hz):
    """核心逻辑：根据 C++ 定义的时间窗口进行物理切分"""
    segments = []
    t_arr = aligned_data['time']
    
    for start_t, end_t, tag in CPP_PHASES:
        # 匹配时间索引
        mask = (t_arr >= start_t) & (t_arr < end_t)
        if not np.any(mask):
            continue
            
        indices = np.where(mask)[0]
        s_idx, e_idx = indices[0], indices[-1]
        
        seg_len = e_idx - s_idx
        if seg_len < target_hz * 2: # 忽略小于 2 秒的碎片
            continue

        # 封装为 Koopman 要求的 (D, N) 格式
        seg_dict = {
            'len': seg_len,
            'Pos': aligned_data['Pos'][s_idx:e_idx].T,
            'Vel': aligned_data['Vel'][s_idx:e_idx].T,
            'pqr': aligned_data['pqr'][s_idx:e_idx].T,
            'Thrusters_CMD': aligned_data['Thrusters_CMD'][s_idx:e_idx].T,
        }
        
        # 欧拉角补全 (3, N)
        euler_3d = np.zeros((3, seg_len), dtype=np.float32)
        euler_3d[2, :] = aligned_data['Euler'][s_idx:e_idx].flatten()
        seg_dict['Euler'] = euler_3d
        
        segments.append(seg_dict)
        print(f"  [切分] 阶段: {tag.ljust(20)} | 持续: {seg_len/target_hz:.1f}s | 点数: {seg_len}")
        
    return segments

def main():
    # --- 运行设置 ---
    bag_files = ["../replay.bag"] 
    output_npz = "koopman_dataset_v1.npz"
    TARGET_HZ = 10.0  # 采样频率

    total_datas = []
    
    for bf in bag_files:
        raw = process_bag(bf)
        if raw is None: continue
        
        aligned = align_and_resample(raw, target_hz=TARGET_HZ)
        if aligned:
            print(f">>> 正在根据 C++ 采样逻辑切分数据片段...")
            segments = split_by_logic(aligned, TARGET_HZ)
            total_datas.extend(segments)
    
    if total_datas:
        # 保存为 Koopman 脚本识别的 object array 格式
        np.savez_compressed(output_npz, datas=np.array(total_datas, dtype=object))
        print(f"\n[完成] 成功提取 {len(total_datas)} 个阶段数据并保存至 {output_npz}")
    else:
        print("\n[错误] 未能提取到任何有效数据，请检查 Bag 时间戳与 C++ 逻辑是否匹配。")

if __name__ == "__main__":
    main()