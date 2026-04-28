import numpy as np

def transform_to_local_frame(current_state, next_state_global):
    """
    将下一时刻的全局状态转换到当前时刻的局部坐标系下
    
    参数:
    current_state: [x_t, y_t, psi_t, u_t, v_t, omega_t]
    next_state_global: [x_{t+1}, y_{t+1}, psi_{t+1}, u_{t+1}, v_{t+1}, omega_{t+1}]
    
    返回:
    local_state_next: 在 t 时刻坐标系下的状态
    """
    # 提取当前时刻位姿
    xt, yt, psit = current_state[0:3]
    
    # 提取下一时刻全局位姿
    xt1, yt1, psit1 = next_state_global[0:3]
    ut1, vt1, omegat1 = next_state_global[3:6]

    # 1. 位置平移
    dx = xt1 - xt
    dy = yt1 - yt

    # 2. 旋转变换 (全局 -> 局部)
    # 使用旋转矩阵的逆，即乘以 [-psit] 的旋转矩阵
    cos_psi = np.cos(psit)
    sin_psi = np.sin(psit)
    
    local_x = dx * cos_psi + dy * sin_psi
    local_y = -dx * sin_psi + dy * cos_psi
    
    # 3. 角度变换 (保证在 [-pi, pi])
    local_psi = psit1 - psit
    local_psi = (local_psi + np.pi) % (2 * np.pi) - np.pi

    # 4. 速度与角速度
    # u, v, omega 通常定义在车辆自身中心，已经是相对量。
    # 如果 next_state_global 里的 u, v 是相对于 t+1 时刻车头的，
    # 那么在 t 时刻局部系下，它们通常保持物理意义不变（即仍为纵向/侧向速度）。
    local_u = ut1
    local_v = vt1
    local_omega = omegat1

    return np.array([local_x, local_y, local_psi, local_u, local_v, local_omega])

# --- 测试代码 ---
s_t = [100.0, 50.0, np.pi/4, 10.0, 0.1, 0.02]  # 当前在 (100, 50)，航向 45度
s_t1_global = [107.0, 57.0, np.pi/4 + 0.05, 10.5, 0.1, 0.02] # 下一时刻全局位置

s_t1_local = transform_to_local_frame(s_t, s_t1_global)

print("下一时刻在当前坐标系下的相对状态:")
print(f"x: {s_t1_local[0]:.3f} m")
print(f"y: {s_t1_local[1]:.3f} m")
print(f"psi (relative): {s_t1_local[2]:.3f} rad")
print(f"u: {s_t1_local[3]:.3f} m/s")
print(f"v: {s_t1_local[4]:.3f} m/s")
print(f"omega: {s_t1_local[5]:.3f} rad/s")