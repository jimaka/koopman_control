#pragma once

/**
 * @file koopman_onnx_model.hpp
 * @brief Koopman v4 动力学 ONNX 推理封装
 *
 * 接口与 Python export_v4_onnx.py 导出图一致：
 *   输入  state0(6), u_seq(H,4), dt
 *   输出  states(H+1, 6)
 */

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace Ort {
struct Session;
struct Env;
struct SessionOptions;
}  // namespace Ort

namespace koopman_control {

/** 基于 ONNX Runtime 的开环 rollout 预测器 */
class KoopmanOnnxModel {
public:
    /** @param onnx_path koopman_rollout.onnx 路径 */
    explicit KoopmanOnnxModel(const std::string& onnx_path);
    ~KoopmanOnnxModel();

    KoopmanOnnxModel(const KoopmanOnnxModel&) = delete;
    KoopmanOnnxModel& operator=(const KoopmanOnnxModel&) = delete;
    KoopmanOnnxModel(KoopmanOnnxModel&&) noexcept;
    KoopmanOnnxModel& operator=(KoopmanOnnxModel&&) noexcept;

    /** 从 ONNX 输入 u_seq 的 shape[0] 读取 H */
    int horizon() const { return horizon_; }

    /**
     * @brief 开环 rollout 一步（整段 H 步）
     * @param state0 初始状态 [x,y,yaw,u,v,r]
     * @param u_seq_flat 长度 H*4 的控制序列（行主序）
     * @param dt 积分步长
     * @return 长度 (H+1)*6 的状态序列（行主序）
     */
    std::vector<float> rollout(const std::array<float, 6>& state0,
                               const std::vector<float>& u_seq_flat,
                               float dt) const;

private:
    /** 解析 ONNX 图中 u_seq 输入维度 */
    int readHorizonFromSession() const;

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> options_;
    std::unique_ptr<Ort::Session> session_;
    int horizon_{0};
};

}  // namespace koopman_control
