import rosbag
import numpy as np
from scipy.interpolate import interp1d

TOPIC_POSE = '/localization/fusion_pose'
TOPIC_THRUSTER = '/system/chassis_feedback'

def generate_exact_phases():
    """针对 15000s 数据包按比例重新调整的物理分段边界"""
    train_phases, val_phases, test_phases = [], [], []
    
    # === 训练集 (0 - 11000s) ===
    for t in range(0, 600, 60): train_phases.append((t, t + 60.0))       # Fwd (直航)
    for t in range(600, 1000, 100): train_phases.append((t, t + 100.0))  # Astern (倒车)
    for t in range(1000, 2000, 200): train_phases.append((t, t + 200.0)) # Diff Turn (差速转向)
    for t in range(2000, 9000, 200): train_phases.append((t, t + 200.0)) # Zigzag (Z字操舵)
    train_phases.append((9000.0, 10000.0))  # Chirp (扫频信号)
    train_phases.append((10000.0, 11000.0)) # PRBS (伪随机二进制序列)
    
    # === 验证集 (11000s - 13000s) ===
    val_phases.append((11000.0, 11700.0)) # Fig-8 (8字航行)
    val_phases.append((11700.0, 12400.0)) # Crash Stop (急停)
    val_phases.append((12400.0, 13000.0)) # U-Turn (U型转弯)
    
    # === 测试集 (13000s - 15000s) ===
    # 随机航行按 200s 切段，方便数据读取
    for t in range(13000, 15000, 200): test_phases.append((t, t + 200.0))
        
    return train_phases, val_phases, test_phases


def process_and_split(bag_path, target_hz=10.0):
    raw_data = {'odom_ts': [], 'Pos': [], 'Vel': [], 'Yaw': [], 'pqr': [], 'cmd_ts': [], 'Thrusters_CMD': []}
    print(f">>> 正在读取 15000s (约4.1小时) 全包络高密度 Bag: {bag_path}")
    
    bag = rosbag.Bag(bag_path)
    for topic, msg, t in bag.read_messages(topics=[TOPIC_POSE, TOPIC_THRUSTER]):
        if topic == TOPIC_POSE:
            raw_data['odom_ts'].append(t.to_sec())
            raw_data['Pos'].append([msg.position.x, msg.position.y])
            raw_data['Vel'].append([msg.velocity.x, msg.velocity.y])
            raw_data['pqr'].append([msg.angular_velocity.z]) 
            raw_data['Yaw'].append(msg.rpy.yaw)
        elif topic == TOPIC_THRUSTER:
            raw_data['cmd_ts'].append(t.to_sec())
            raw_data['Thrusters_CMD'].append([msg.port_thruster_throttle, msg.port_thruster_angle, msg.starboard_thruster_throttle, msg.starboard_thruster_angle])
    bag.close()

    ts_odom, ts_cmd = np.array(raw_data['odom_ts']), np.array(raw_data['cmd_ts'])
    t0 = min(ts_odom[0], ts_cmd[0])
    ts_odom -= t0; ts_cmd -= t0
    t_common = np.arange(max(ts_odom[0], ts_cmd[0]), min(ts_odom[-1], ts_cmd[-1]), 1.0 / target_hz)
    
    aligned = {'time': t_common}
    aligned['Pos'] = interp1d(ts_odom, raw_data['Pos'], axis=0, fill_value="extrapolate")(t_common).astype(np.float32)
    aligned['Vel'] = interp1d(ts_odom, raw_data['Vel'], axis=0, fill_value="extrapolate")(t_common).astype(np.float32)
    aligned['pqr'] = interp1d(ts_odom, raw_data['pqr'], axis=0, fill_value="extrapolate")(t_common).astype(np.float32)
    aligned['Euler'] = interp1d(ts_odom, np.unwrap(raw_data['Yaw']), fill_value="extrapolate")(t_common).reshape(-1, 1).astype(np.float32)
    aligned['Thrusters_CMD'] = interp1d(ts_cmd, raw_data['Thrusters_CMD'], axis=0, fill_value="extrapolate")(t_common).astype(np.float32)

    def extract_segments(time_ranges):
        segs = []
        for start_t, end_t in time_ranges:
            mask = (t_common >= start_t) & (t_common < end_t)
            if not np.any(mask): continue
            indices = np.where(mask)[0]
            # 过滤掉不足 20 秒 (20 * target_hz 帧) 的小碎片
            if indices[-1] - indices[0] < target_hz * 20: continue 

            seg_dict = {
                'len': indices[-1] - indices[0],
                'Pos': aligned['Pos'][indices[0]:indices[-1]].T,
                'Vel': aligned['Vel'][indices[0]:indices[-1]].T,
                'pqr': aligned['pqr'][indices[0]:indices[-1]].T,
                'Thrusters_CMD': aligned['Thrusters_CMD'][indices[0]:indices[-1]].T,
            }
            euler_3d = np.zeros((3, seg_dict['len']), dtype=np.float32)
            euler_3d[2, :] = aligned['Euler'][indices[0]:indices[-1]].flatten()
            seg_dict['Euler'] = euler_3d
            segs.append(seg_dict)
        return segs

    train_bounds, val_bounds, test_bounds = generate_exact_phases()
    
    train_segs = extract_segments(train_bounds)
    val_segs   = extract_segments(val_bounds)
    test_segs  = extract_segments(test_bounds)

    np.savez_compressed("koopman_train.npz", datas=np.array(train_segs, dtype=object))
    np.savez_compressed("koopman_val.npz", datas=np.array(val_segs, dtype=object))
    np.savez_compressed("koopman_test.npz", datas=np.array(test_segs, dtype=object))
    
    print(f"\n✅ 15000s 数据集分离完成！")
    print(f"  - 训练集 (高密网格辨识): {len(train_segs)} 段")
    print(f"  - 验证集 (标准海试机动): {len(val_segs)} 段")
    print(f"  - 测试集 (自由随机盲考): {len(test_segs)} 段")

if __name__ == "__main__":
    process_and_split("../replay.bag")