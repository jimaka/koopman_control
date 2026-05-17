import rosbag
import numpy as np
from scipy.interpolate import interp1d

TOPIC_POSE = '/localization/fusion_pose'
TOPIC_THRUSTER = '/system/chassis_feedback'

def generate_exact_phases():
    """根据高密度 C++ 脚本精准生成物理分段边界"""
    train_phases, val_phases, test_phases = [], [], []
    
    # === 训练集 (0 - 18000s) ===
    for t in range(0, 900, 90): train_phases.append((t, t + 90.0))       # Fwd
    for t in range(900, 1500, 100): train_phases.append((t, t + 100.0))  # Astern
    for t in range(1500, 3300, 200): train_phases.append((t, t + 200.0)) # Diff Turn
    for t in range(3300, 15900, 200): train_phases.append((t, t + 200.0))# Zigzag 63组
    train_phases.append((15900.0, 16900.0)) # Chirp
    train_phases.append((16900.0, 18000.0)) # PRBS
    
    # === 验证集 (18000s - 21600s) ===
    val_phases.append((18000.0, 19200.0)) # Fig-8
    val_phases.append((19200.0, 20400.0)) # Crash Stop
    val_phases.append((20400.0, 21600.0)) # U-Turn
    
    # === 测试集 (21600s - 25200s) ===
    # 随机航行也按 200s 切段，方便数据读取
    for t in range(21600, 25200, 200): test_phases.append((t, t + 200.0))
        
    return train_phases, val_phases, test_phases


def process_and_split(bag_path, target_hz=10.0):
    raw_data = {'odom_ts': [], 'Pos': [], 'Vel': [], 'Yaw': [], 'pqr': [], 'cmd_ts': [], 'Thrusters_CMD': []}
    print(f">>> 正在读取 7小时全包络高密度 Bag: {bag_path}")
    
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

    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from koopman import paths as P

    np.savez_compressed(str(P.TRAIN), datas=np.array(train_segs, dtype=object))
    np.savez_compressed(str(P.VAL), datas=np.array(val_segs, dtype=object))
    np.savez_compressed(str(P.TEST), datas=np.array(test_segs, dtype=object))
    
    print(f"\n✅ 高密度数据集分离完成！")
    print(f"  - 训练集 (高密网格辨识): {len(train_segs)} 段")
    print(f"  - 验证集 (标准海试机动): {len(val_segs)} 段")
    print(f"  - 测试集 (自由随机盲考): {len(test_segs)} 段")

if __name__ == "__main__":
    process_and_split("../replay.bag")