#pragma once

/**
 * @file mpc_controller.hpp
 * @brief 基于 Koopman ONNX 模型的滚动时域 MPC 求解器
 *
 * 优化方法：Adam + 前向差分数值梯度（无 autograd）
 * 每步仅返回控制序列 u[0]（receding horizon）
 */

#include <array>
#include <utility>
#include <vector>

#include "koopman_control/koopman_onnx_model.hpp"
#include "koopman_control/mpc_config.hpp"

namespace koopman_control {

/** 闭环仿真记录 */
struct MpcTrajectory {
    std::vector<float> t;                          ///< 时间戳
    std::vector<std::array<float, 6>> state;         ///< 实际状态轨迹
    std::vector<std::array<float, 4>> control;     ///< 每步 applied u0
    std::vector<std::array<float, 6>> ref_state;   ///< 参考状态
    std::vector<float> cost_history;               ///< 每步 MPC 最优代价
};

/** 航迹跟踪误差统计 */
struct TrackingMetrics {
    float xy_rmse_m = 0.f;      ///< 平面位置 RMSE (m)
    float xy_max_m = 0.f;       ///< 平面最大偏差 (m)
    float yaw_rmse_deg = 0.f;   ///< 航向 RMSE (deg)
    float final_xy_err_m = 0.f; ///< 终点平面误差 (m)
};

/** Koopman MPC 主控制器 */
class KoopmanMpcController {
public:
    KoopmanMpcController(KoopmanOnnxModel model, MpcConfig cfg);

    /**
     * @brief 求解单步 MPC，返回首步控制 u0
     * @param state0 当前状态 [x,y,yaw,u,v,r]
     * @param ref_window 长度 horizon+1 的参考状态窗口
     * @return {u0, 最优代价}
     */
    std::pair<std::array<float, 4>, float> solveStep(
        const std::array<float, 6>& state0,
        const std::vector<std::array<float, 6>>& ref_window,
        const std::array<float, 4>* u_prev_applied = nullptr);

    /**
     * @brief 沿参考航迹闭环仿真
     * @param ref_ctrl 可选：用于 warm-start 的参考控制序列
     */
    MpcTrajectory simulate(const std::array<float, 6>& state0,
                           const std::vector<std::array<float, 6>>& ref_traj,
                           const std::vector<std::array<float, 4>>* ref_ctrl,
                           int max_steps);

    int horizon() const { return cfg_.horizon; }
    const MpcConfig& config() const { return cfg_; }
    /** 清除上一步 warm-start 缓存 */
    void resetWarmStart();

private:
    /** 计算 MPC 代价（跟踪 + 控制正则） */
    float mpcCost(const std::array<float, 6>& state0,
                  const std::vector<std::array<float, 6>>& ref,
                  const std::vector<float>& u_flat,
                  const std::array<float, 4>& u_prev) const;

    /** 对前 opt_control_blocks*4 维控制块做数值梯度 */
    std::vector<float> numericGrad(const std::array<float, 6>& state0,
                                   const std::vector<std::array<float, 6>>& ref,
                                   std::vector<float> u_blocks,
                                   const std::array<float, 4>& u_prev) const;

    int controlHoldSteps() const;
    int numControlBlocks() const;
    int optControlBlocks() const;
    void expandBlocksToFlat(const std::vector<float>& u_blocks,
                             std::vector<float>& u_flat) const;
    void extractBlocksFromFlat(const std::vector<float>& u_flat,
                               std::vector<float>& u_blocks) const;
    void enforceBlockingOnFlat(std::vector<float>& u_flat) const;
    void clampBlocks(std::vector<float>& u_blocks,
                     const std::array<float, 4>& u_prev_step0) const;
    void fillHoldBlocks(std::vector<float>& u_blocks) const;
    /** 裁剪控制块、展开为 u_flat（供 rollout / 存储） */
    void finalizeBlocks(std::vector<float>& u_blocks,
                        const std::array<float, 4>& u_prev_step0,
                        std::vector<float>& u_flat) const;
    /** 单通道有效变化速率上限（块间 transition） */
    float effectiveDuMax(int channel) const;
    /** 单通道增量软惩罚权重 */
    float duWeight(int channel) const;

    KoopmanOnnxModel model_;
    MpcConfig cfg_;
    std::vector<float> u_warm_;       ///< 上一步最优控制序列，用于 warm-start
    bool has_warm_{false};
    std::array<float, 4> u_applied_{};  ///< 上一步实际下发的 u0
    bool has_applied_{false};
};

/** 由仿真轨迹计算跟踪指标 */
TrackingMetrics computeMetrics(const MpcTrajectory& traj);

}  // namespace koopman_control
