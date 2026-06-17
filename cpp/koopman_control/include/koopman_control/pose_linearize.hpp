#pragma once

/**
 * @file pose_linearize.hpp
 * @brief Tier-2：在标称轨迹处线性化 decoder + 船体系欧拉积分，
 *        得到位姿 (x,y,yaw) 对控制序列 U 的仿射灵敏度 Phi 与偏置 b。
 *
 * 预测位姿近似： P_pose ≈ p_free + Phi·U，其中
 *   cost = Σ_k wq_k · (Phi·U + b)_k^2,  b = nominal_err − Phi·U0（yaw 已 wrap）。
 */

#include <array>
#include <vector>

#include "koopman_control/detail/dense_matrix.hpp"

namespace koopman_control {

class KoopmanLatentModel;
class KoopmanDecoder;

struct PoseLinearization {
    detail::Matrix Phi;       // (3N) x (nu*N)
    std::vector<float> b;     // (3N)
    std::vector<float> wq;    // (3N) 对角权重：x,y -> w_xy；yaw -> w_yaw
    bool valid{false};
};

/**
 * @param z0          当前潜变量 (nz,)
 * @param pose0       初始位姿 [x,y,yaw]（与 ref 同坐标系）
 * @param u_tilde_U0  标称归一化控制序列 (nu*N,)
 * @param pose_ref    参考位姿堆叠 (3N,)，对应步 k=1..N 的 [x,y,yaw]
 */
PoseLinearization buildPoseLinearization(const KoopmanLatentModel& model,
                                         const KoopmanDecoder& decoder,
                                         const std::vector<float>& z0,
                                         const std::array<float, 3>& pose0,
                                         const std::vector<float>& u_tilde_U0,
                                         const std::vector<float>& pose_ref,
                                         float dt, float w_xy, float w_yaw);

}  // namespace koopman_control
