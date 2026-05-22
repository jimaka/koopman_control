#pragma once

/**
 * @file mpc_config.hpp
 * @brief Koopman MPC 控制器配置参数
 *
 * 状态向量：[x, y, yaw, u, v, r]（船体坐标系）
 * 控制向量：4 维，与训练数据集 ctrl 字段一致
 */

#include <array>

namespace koopman_control {

/** MPC 代价权重、优化器与约束配置 */
struct MpcConfig {
    /** 预测步数 H，须与 ONNX 导出 pred_len 一致（v4 20s @ dt=1.0 为 20） */
    int horizon = 20;
    /** 仅优化前 N 步控制；后续步零阶保持，降低长 horizon 下数值梯度开销 */
    int opt_control_steps = 2;
    /** 离散时间步长（秒），须与训练/ONNX 一致 */
    float dt = 1.0f;
    /**
     * 控制块长度（细步数）：每 hold 步共用一个控制量。
     * dt=1.0 时通常设为 1。
     */
    int control_hold_steps = 1;
    /** 位置跟踪权重 (x,y) */
    float w_xy = 10.f;
    /** 航向跟踪权重 yaw */
    float w_yaw = 5.f;
    /** 速度跟踪权重 (u,v,r) */
    float w_vel = 0.5f;
    /** 控制量幅值惩罚 */
    float w_u = 1e-4f;
    /** 控制增量惩罚，抑制抖动（全局默认） */
    float w_du = 0.05f;
    /** 油门通道 (0,2) 增量惩罚；<0 时使用 w_du */
    float w_du_throttle = -1.f;
    /** 舵角通道 (1,3) 增量惩罚；<0 时使用 w_du */
    float w_du_rudder = -1.f;
    /**
     * 每步最大控制变化量（硬约束，单位与 u 一致）。
     * du_max[j] > 0 时对该通道生效；<=0 时回退到 throttle_du_max / rudder_du_max。
     */
    std::array<float, 4> du_max{0.f, 0.f, 0.f, 0.f};
    /** 油门通道 (0,2) 每步最大变化；<=0 表示不限制 */
    float throttle_du_max = 0.f;
    /** 舵角通道 (1,3) 每步最大变化；<=0 表示不限制 */
    float rudder_du_max = 0.f;
    /** Adam 优化迭代次数 */
    int opt_iters = 8;
    /** Adam 学习率 */
    float opt_lr = 0.08f;
    /** 4 维控制下界 */
    std::array<float, 4> u_min{-100.f, -35.f, -100.f, -35.f};
    /** 4 维控制上界 */
    std::array<float, 4> u_max{100.f, 35.f, 100.f, 35.f};
};

}  // namespace koopman_control
