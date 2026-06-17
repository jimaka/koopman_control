/**
 * @file koopman_decoder.cpp
 */

#include "koopman_control/koopman_decoder.hpp"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

namespace koopman_control {
namespace {

struct LinearW {
    std::vector<float> weight;  // row-major (out x in)
    std::vector<float> bias;
    int in_dim{0};
    int out_dim{0};
};

enum class LayerType { kLinear, kGelu };

struct DecoderLayer {
    LayerType type;
    LinearW lin;  // valid when type==kLinear
};

LinearW loadLinearNode(const YAML::Node& node) {
    LinearW l;
    const auto& w = node["weight"];
    l.out_dim = static_cast<int>(w.size());
    l.in_dim = l.out_dim > 0 ? static_cast<int>(w[0].size()) : 0;
    l.weight.assign(static_cast<size_t>(l.out_dim * l.in_dim), 0.f);
    for (int r = 0; r < l.out_dim; ++r) {
        for (int c = 0; c < l.in_dim; ++c) {
            l.weight[static_cast<size_t>(r * l.in_dim + c)] = w[r][c].as<float>();
        }
    }
    for (const auto& b : node["bias"]) {
        l.bias.push_back(b.as<float>());
    }
    return l;
}

}  // namespace

struct KoopmanDecoder::Impl {
    std::vector<DecoderLayer> layers;
    std::array<float, 3> dyn_mean{};
    std::array<float, 3> dyn_std{1.f, 1.f, 1.f};
    bool ok{false};
};

KoopmanDecoder::KoopmanDecoder() = default;
KoopmanDecoder::~KoopmanDecoder() = default;
KoopmanDecoder::KoopmanDecoder(KoopmanDecoder&&) noexcept = default;
KoopmanDecoder& KoopmanDecoder::operator=(KoopmanDecoder&&) noexcept = default;

void KoopmanDecoder::loadFromYaml(const std::string& yaml_path) {
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    YAML::Node root = YAML::LoadFile(yaml_path);
    if (!root["decoder"] || !root["decoder"]["layers"]) {
        impl_->ok = false;
        return;  // 旧 YAML 无 decoder：Tier-2 不可用，但不报错
    }

    const auto norm = root["normalization"];
    for (int i = 0; i < 3; ++i) {
        impl_->dyn_mean[static_cast<size_t>(i)] = norm["dyn_mean"][i].as<float>();
        impl_->dyn_std[static_cast<size_t>(i)] = norm["dyn_std"][i].as<float>();
    }

    impl_->layers.clear();
    for (const auto& layer : root["decoder"]["layers"]) {
        const std::string type = layer["type"].as<std::string>();
        DecoderLayer dl;
        if (type == "linear") {
            dl.type = LayerType::kLinear;
            dl.lin = loadLinearNode(layer);
        } else if (type == "gelu") {
            dl.type = LayerType::kGelu;
        } else {
            throw std::runtime_error("unknown decoder layer type: " + type);
        }
        impl_->layers.push_back(std::move(dl));
    }
    impl_->ok = true;
}

bool KoopmanDecoder::loaded() const {
    return impl_ && impl_->ok;
}

std::array<float, 3> KoopmanDecoder::decodePhysical(const std::vector<float>& z) const {
    if (!loaded()) {
        throw std::runtime_error("KoopmanDecoder not loaded");
    }
    std::vector<float> h = z;
    for (const auto& dl : impl_->layers) {
        if (dl.type == LayerType::kLinear) {
            std::vector<float> y(static_cast<size_t>(dl.lin.out_dim), 0.f);
            for (int r = 0; r < dl.lin.out_dim; ++r) {
                float s = dl.lin.bias[static_cast<size_t>(r)];
                for (int c = 0; c < dl.lin.in_dim; ++c) {
                    s += dl.lin.weight[static_cast<size_t>(r * dl.lin.in_dim + c)] *
                         h[static_cast<size_t>(c)];
                }
                y[static_cast<size_t>(r)] = s;
            }
            h = std::move(y);
        } else {
            for (float& v : h) {
                v = detail::gelu(v);
            }
        }
    }
    std::array<float, 3> out{};
    for (int i = 0; i < 3; ++i) {
        out[static_cast<size_t>(i)] =
            h[static_cast<size_t>(i)] * impl_->dyn_std[static_cast<size_t>(i)] +
            impl_->dyn_mean[static_cast<size_t>(i)];
    }
    return out;
}

detail::Matrix KoopmanDecoder::jacobianPhysical(const std::vector<float>& z) const {
    if (!loaded()) {
        throw std::runtime_error("KoopmanDecoder not loaded");
    }
    const int nz = static_cast<int>(z.size());

    // 前向缓存每个 Linear 的输入，以及每个 GELU 的输入（用于导数）。
    // J = (∏ 各层局部雅可比) ，逐层从输入侧累乘：J_running (cur_dim x nz)
    detail::Matrix J = detail::Matrix::identity(nz);  // d h / d z，初始 h=z
    std::vector<float> h = z;

    for (const auto& dl : impl_->layers) {
        if (dl.type == LayerType::kLinear) {
            const int out_dim = dl.lin.out_dim;
            const int in_dim = dl.lin.in_dim;
            detail::Matrix W(out_dim, in_dim, 0.f);
            for (int r = 0; r < out_dim; ++r) {
                for (int c = 0; c < in_dim; ++c) {
                    W(r, c) = dl.lin.weight[static_cast<size_t>(r * in_dim + c)];
                }
            }
            J = detail::Matrix::matmul(W, J);  // (out x nz)
            // 更新 h
            std::vector<float> y(static_cast<size_t>(out_dim), 0.f);
            for (int r = 0; r < out_dim; ++r) {
                float s = dl.lin.bias[static_cast<size_t>(r)];
                for (int c = 0; c < in_dim; ++c) {
                    s += W(r, c) * h[static_cast<size_t>(c)];
                }
                y[static_cast<size_t>(r)] = s;
            }
            h = std::move(y);
        } else {
            // GELU：逐元素，局部雅可比为对角 diag(gelu'(h_i))
            const int dim = static_cast<int>(h.size());
            for (int r = 0; r < dim; ++r) {
                const float g = detail::geluGrad(h[static_cast<size_t>(r)]);
                for (int c = 0; c < nz; ++c) {
                    J(r, c) *= g;
                }
                h[static_cast<size_t>(r)] = detail::gelu(h[static_cast<size_t>(r)]);
            }
        }
    }

    // 物理缩放：J_phys = diag(dyn_std) · J_norm（J 现为 3 x nz）
    detail::Matrix Jp(3, nz, 0.f);
    for (int r = 0; r < 3; ++r) {
        const float s = impl_->dyn_std[static_cast<size_t>(r)];
        for (int c = 0; c < nz; ++c) {
            Jp(r, c) = s * J(r, c);
        }
    }
    return Jp;
}

}  // namespace koopman_control
