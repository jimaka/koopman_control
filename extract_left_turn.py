import rosbag
import numpy as np
import math
from scipy.interpolate import interp1d

TOPIC_POSE = '/localization/fusion_pose'
TOPIC_THRUSTER = '/system/chassis_feedback'

def process_single_bag(bag_path, output_npz, segment_length=200.0, target_hz=10.0):
    raw_data = {'odom_ts': [], 'Pos': [], 'Vel': [], 'Yaw': [], 'pqr': [], 'cmd_ts': [], 'Thrusters_CMD': []}
    print(f">>> 正在读取补充采集的 Bag: {bag_path}")
    
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

    # 1. 时间戳对齐与插值
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

    # 2. 自动按指定时长切分（默认 200s）
    segs = []
    total_duration = t_common[-1] - t_common[0]
    
    # 【修复】: 使用向上取整，确保由于录包时间误差导致的最后不足 200s 的长尾数据不被丢弃
    num_segments = math.ceil(total_duration / segment_length)
    
    for i in range(num_segments):
        start_t = i * segment_length
        end_t = (i + 1) * segment_length
        
        mask = (t_common >= start_t) & (t_common < end_t)
        if not np.any(mask): continue
        indices = np.where(mask)[0]
        
        # 确保数据段足够长：只要达到目标段长度的 90% (即 > 180秒) 就保留
        if indices[-1] - indices[0] < target_hz * (segment_length * 0.9): 
            print(f"⚠️ 忽略尾部过短的数据段 (第 {i+1} 段): 时长仅 {(indices[-1] - indices[0])/target_hz:.1f} 秒")
            continue 

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

    # 3. 保存为独立的 npz
    np.savez_compressed(output_npz, datas=np.array(segs, dtype=object))
    print(f"✅ 提取完成！共提取 {len(segs)} 段 {segment_length}s 的数据，保存至: {output_npz}")

if __name__ == "__main__":
    process_single_bag("../replay.bag", "koopman_train_left_turn.npz")