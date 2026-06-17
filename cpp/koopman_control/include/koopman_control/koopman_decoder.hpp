#pragma once

/**
 * @file koopman_decoder.hpp
 * @brief v4 decoder（z(48) -> 物理 [u,v,r]）前向 + Jacobian，供 Tier-2 位姿线性化。
 */

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "koopman_control/detail/dense_matrix.hpp"

namespace koopman_control {

class KoopmanDecoder {
public:
    KoopmanDecoder();
    ~KoopmanDecoder();
    KoopmanDecoder(KoopmanDecoder&&) noexcept;
    KoopmanDecoder& operator=(KoopmanDecoder&&) noexcept;
    KoopmanDecoder(const KoopmanDecoder&) = delete;
    KoopmanDecoder& operator=(const KoopmanDecoder&) = delete;

    /** 读取 decoder.layers 与 normalization.dyn_mean/std */
    void loadFromYaml(const std::string& yaml_path);

    bool loaded() const;

    /** z(nz,) -> 物理 [u,v,r]（已反归一化） */
    std::array<float, 3> decodePhysical(const std::vector<float>& z) const;

    /** d[u,v,r]_phys / dz，形状 3 x nz（含 dyn_std 缩放） */
    detail::Matrix jacobianPhysical(const std::vector<float>& z) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace koopman_control
