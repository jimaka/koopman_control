#pragma once

/**
 * @file koopman_encode.hpp
 * @brief v4 dict16 + res_mlp encoder（C++ 推理，权重来自 YAML）。
 */

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace koopman_control {

class KoopmanEncoder {
public:
    KoopmanEncoder();
    ~KoopmanEncoder();
    KoopmanEncoder(KoopmanEncoder&&) noexcept;
    KoopmanEncoder& operator=(KoopmanEncoder&&) noexcept;
    KoopmanEncoder(const KoopmanEncoder&) = delete;
    KoopmanEncoder& operator=(const KoopmanEncoder&) = delete;

    void loadFromYaml(const std::string& yaml_path);

    /** dyn_norm: 归一化 [u,v,r] -> z (48,) */
    std::vector<float> encode(const std::array<float, 3>& dyn_norm) const;

    float clampPif() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    static std::vector<float> computeAtoms16(float u, float v, float r, float clamp_pif);
};

}  // namespace koopman_control
