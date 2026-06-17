/**
 * @file koopman_latent_model.cpp
 */

#include "koopman_control/koopman_latent_model.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <stdexcept>

namespace koopman_control {
namespace {

detail::Matrix loadMatrix(const YAML::Node& node) {
    if (!node.IsSequence()) {
        throw std::runtime_error("expected matrix sequence in yaml");
    }
    const int rows = static_cast<int>(node.size());
    if (rows == 0) {
        return detail::Matrix(0, 0);
    }
    const int cols = static_cast<int>(node[0].size());
    detail::Matrix m(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            m(r, c) = node[r][c].as<float>();
        }
    }
    return m;
}

std::vector<float> loadVector(const YAML::Node& node) {
    std::vector<float> v;
    v.reserve(node.size());
    for (const auto& x : node) {
        v.push_back(x.as<float>());
    }
    return v;
}

}  // namespace

void KoopmanLatentModel::loadFromYaml(const std::string& yaml_path, int horizon) {
    YAML::Node root = YAML::LoadFile(yaml_path);
    horizon_ = horizon;
    nz_ = root["latent_dim"] ? root["latent_dim"].as<int>() : LatentDims::NZ;
    nu_ = root["control_dim"] ? root["control_dim"].as<int>() : LatentDims::NU;
    clamp_pif_ = root["clamp_pif"] ? root["clamp_pif"].as<float>() : 5.f;

    const auto norm = root["normalization"];
    for (int i = 0; i < 3; ++i) {
        dyn_mean_[i] = norm["dyn_mean"][i].as<float>();
        dyn_std_[i] = norm["dyn_std"][i].as<float>();
    }
    for (int i = 0; i < 4; ++i) {
        ctrl_mean_[i] = norm["ctrl_mean"][i].as<float>();
        ctrl_std_[i] = norm["ctrl_std"][i].as<float>();
    }

    const auto k = root["koopman"];
    Abar_ = loadMatrix(k["A_bar"]);
    Bbar_ = loadMatrix(k["B"]);
    bias_ = loadVector(k["bias"]);

    if (Abar_.rows() != nz_ || Abar_.cols() != nz_) {
        throw std::runtime_error("A_bar shape mismatch");
    }
    if (Bbar_.rows() != nz_ || Bbar_.cols() != nu_) {
        throw std::runtime_error("B shape mismatch");
    }
    if (static_cast<int>(bias_.size()) != nz_) {
        throw std::runtime_error("bias size mismatch");
    }
}

void KoopmanLatentModel::precomputePredictionMatrices() {
    const int n = horizon_;
    const int nz = nz_;
    const int nu = nu_;

    std::vector<detail::Matrix> A_pow(static_cast<size_t>(n + 1));
    A_pow[0] = detail::Matrix::identity(nz);
    for (int i = 1; i <= n; ++i) {
        A_pow[static_cast<size_t>(i)] = detail::Matrix::matmul(A_pow[static_cast<size_t>(i - 1)], Abar_);
    }

    Gamma_ = detail::Matrix(nz * n, nz, 0.f);
    Theta_ = detail::Matrix(nz * n, nu * n, 0.f);
    xi_stack_.assign(static_cast<size_t>(nz * n), 0.f);

    for (int k = 1; k <= n; ++k) {
        const int row = (k - 1) * nz;
        const detail::Matrix& Ak = A_pow[static_cast<size_t>(k)];
        for (int r = 0; r < nz; ++r) {
            for (int c = 0; c < nz; ++c) {
                Gamma_(row + r, c) = Ak(r, c);
            }
        }

        std::vector<float> xi_k(static_cast<size_t>(nz), 0.f);
        detail::Matrix A_i = detail::Matrix::identity(nz);
        for (int i = 0; i < k; ++i) {
            for (int r = 0; r < nz; ++r) {
                float s = 0.f;
                for (int c = 0; c < nz; ++c) {
                    s += A_i(r, c) * bias_[static_cast<size_t>(c)];
                }
                xi_k[static_cast<size_t>(r)] += s;
            }
            A_i = detail::Matrix::matmul(Abar_, A_i);
        }
        for (int r = 0; r < nz; ++r) {
            xi_stack_[static_cast<size_t>(row + r)] = xi_k[static_cast<size_t>(r)];
        }

        for (int j = 0; j < k; ++j) {
            const int power = k - j - 1;
            const detail::Matrix& A_kj = A_pow[static_cast<size_t>(power)];
            const detail::Matrix block = detail::Matrix::matmul(A_kj, Bbar_);
            for (int r = 0; r < nz; ++r) {
                for (int c = 0; c < nu; ++c) {
                    Theta_(row + r, j * nu + c) = block(r, c);
                }
            }
        }
    }
}

std::vector<float> KoopmanLatentModel::freeResponse(const std::vector<float>& z0) const {
    auto zf = detail::Matrix::matvec(Gamma_, z0);
    return detail::Matrix::add(zf, xi_stack_);
}

std::vector<float> KoopmanLatentModel::predictStacked(const std::vector<float>& z0,
                                                      const std::vector<float>& u_stack) const {
    auto zf = freeResponse(z0);
    auto zu = detail::Matrix::matvec(Theta_, u_stack);
    return detail::Matrix::add(zf, zu);
}

std::vector<float> KoopmanLatentModel::latentStep(const std::vector<float>& z,
                                                  const std::vector<float>& u_tilde) const {
    auto Az = detail::Matrix::matvec(Abar_, z);
    auto Bu = detail::Matrix::matvec(Bbar_, u_tilde);
    std::vector<float> out(z.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = Az[i] + Bu[i] + bias_[i];
    }
    return out;
}

std::vector<float> KoopmanLatentModel::normalizeControl(const std::array<float, 4>& u_phys) const {
    std::vector<float> out(4);
    for (int i = 0; i < 4; ++i) {
        out[static_cast<size_t>(i)] = (u_phys[static_cast<size_t>(i)] - ctrl_mean_[static_cast<size_t>(i)]) /
                                      ctrl_std_[static_cast<size_t>(i)];
    }
    return out;
}

std::array<float, 4> KoopmanLatentModel::denormalizeControl(const std::vector<float>& u_tilde) const {
    std::array<float, 4> out{};
    for (int i = 0; i < 4; ++i) {
        out[static_cast<size_t>(i)] =
            u_tilde[static_cast<size_t>(i)] * ctrl_std_[static_cast<size_t>(i)] + ctrl_mean_[static_cast<size_t>(i)];
    }
    return out;
}

}  // namespace koopman_control
