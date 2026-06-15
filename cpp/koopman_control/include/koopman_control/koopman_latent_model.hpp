#pragma once

/**
 * @file koopman_latent_model.hpp
 * @brief v4 Koopman 潜空间矩阵加载与 condensed 预测 (Gamma, Theta, xi)。
 */

#include <array>
#include <string>
#include <vector>

#include "koopman_control/detail/dense_matrix.hpp"

namespace koopman_control {

struct LatentDims {
    static constexpr int NZ = 48;
    static constexpr int NU = 4;
};

class KoopmanLatentModel {
public:
    int horizon() const { return horizon_; }
    int nz() const { return nz_; }
    int nu() const { return nu_; }

    const detail::Matrix& Abar() const { return Abar_; }
    const detail::Matrix& Bbar() const { return Bbar_; }
    const std::vector<float>& bias() const { return bias_; }
    const std::array<float, 3>& dynMean() const { return dyn_mean_; }
    const std::array<float, 3>& dynStd() const { return dyn_std_; }
    const std::array<float, 4>& ctrlMean() const { return ctrl_mean_; }
    const std::array<float, 4>& ctrlStd() const { return ctrl_std_; }
    float clampPif() const { return clamp_pif_; }

    const detail::Matrix& Gamma() const { return Gamma_; }
    const detail::Matrix& Theta() const { return Theta_; }
    const std::vector<float>& xiStack() const { return xi_stack_; }

    void loadFromYaml(const std::string& yaml_path, int horizon);
    void precomputePredictionMatrices();

    std::vector<float> freeResponse(const std::vector<float>& z0) const;
    std::vector<float> predictStacked(const std::vector<float>& z0, const std::vector<float>& u_stack) const;
    std::vector<float> latentStep(const std::vector<float>& z, const std::vector<float>& u_tilde) const;

    std::vector<float> normalizeControl(const std::array<float, 4>& u_phys) const;
    std::array<float, 4> denormalizeControl(const std::vector<float>& u_tilde) const;

private:
    int horizon_{20};
    int nz_{LatentDims::NZ};
    int nu_{LatentDims::NU};
    float clamp_pif_{5.f};

    detail::Matrix Abar_;
    detail::Matrix Bbar_;
    std::vector<float> bias_;

    std::array<float, 3> dyn_mean_{};
    std::array<float, 3> dyn_std_{};
    std::array<float, 4> ctrl_mean_{};
    std::array<float, 4> ctrl_std_{};

    detail::Matrix Gamma_;
    detail::Matrix Theta_;
    std::vector<float> xi_stack_;
};

}  // namespace koopman_control
