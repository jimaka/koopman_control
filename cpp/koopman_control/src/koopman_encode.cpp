/**
 * @file koopman_encode.cpp
 */

#include "koopman_control/koopman_encode.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <stdexcept>

#include "koopman_control/detail/dense_matrix.hpp"

namespace koopman_control {
namespace {

struct LinearW {
    std::vector<float> weight;
    std::vector<float> bias;
    int in_dim{0};
    int out_dim{0};
};

struct ResidualBlockW {
    LinearW fc;
    std::vector<float> conv_weight;
    std::vector<float> conv_bias;
    LinearW shortcut;
    bool has_shortcut{false};
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
    l.bias.reserve(node["bias"].size());
    for (const auto& b : node["bias"]) {
        l.bias.push_back(b.as<float>());
    }
    return l;
}

std::vector<float> linearForward(const LinearW& layer, const std::vector<float>& x) {
    std::vector<float> y(static_cast<size_t>(layer.out_dim), 0.f);
    for (int r = 0; r < layer.out_dim; ++r) {
        float s = layer.bias[static_cast<size_t>(r)];
        for (int c = 0; c < layer.in_dim; ++c) {
            s += layer.weight[static_cast<size_t>(r * layer.in_dim + c)] * x[static_cast<size_t>(c)];
        }
        y[static_cast<size_t>(r)] = s;
    }
    return y;
}

}  // namespace

struct KoopmanEncoder::Impl {
    float clamp_pif{5.f};
    std::vector<ResidualBlockW> res_blocks;
    LinearW out_linear;
};

void KoopmanEncoder::loadFromYaml(const std::string& yaml_path) {
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    YAML::Node root = YAML::LoadFile(yaml_path);
    impl_->clamp_pif = root["clamp_pif"] ? root["clamp_pif"].as<float>() : 5.f;
    impl_->res_blocks.clear();

    for (const auto& layer : root["encoder"]["layers"]) {
        const std::string type = layer["type"].as<std::string>();
        if (type == "residual_conv_block") {
            ResidualBlockW blk;
            blk.fc = loadLinearNode(layer["fc"]);
            blk.conv_weight.clear();
            for (const auto& row : layer["conv_weight"]) {
                for (const auto& v : row) {
                    blk.conv_weight.push_back(v.as<float>());
                }
            }
            if (layer["conv_bias"]) {
                for (const auto& b : layer["conv_bias"]) {
                    blk.conv_bias.push_back(b.as<float>());
                }
            }
            if (layer["shortcut"] && !layer["shortcut"].IsNull()) {
                blk.shortcut = loadLinearNode(layer["shortcut"]);
                blk.has_shortcut = true;
            }
            impl_->res_blocks.push_back(std::move(blk));
        } else if (type == "linear") {
            impl_->out_linear = loadLinearNode(layer);
        } else {
            throw std::runtime_error("unknown encoder layer type: " + type);
        }
    }
}

KoopmanEncoder::~KoopmanEncoder() = default;
KoopmanEncoder::KoopmanEncoder() = default;
KoopmanEncoder::KoopmanEncoder(KoopmanEncoder&&) noexcept = default;
KoopmanEncoder& KoopmanEncoder::operator=(KoopmanEncoder&&) noexcept = default;

float KoopmanEncoder::clampPif() const {
    return impl_ ? impl_->clamp_pif : 5.f;
}

std::vector<float> KoopmanEncoder::computeAtoms16(float u, float v, float r, float clamp_pif) {
    const float abs_u = std::fabs(u);
    const float abs_v = std::fabs(v);
    const float abs_r = std::fabs(r);
    std::vector<float> atoms(16);
    atoms[0] = u * abs_u;
    atoms[1] = v * abs_v;
    atoms[2] = r * abs_r;
    atoms[3] = v * r;
    atoms[4] = u * r;
    atoms[5] = u * v * r;
    atoms[6] = u * u * r;
    atoms[7] = v * v * r;
    atoms[8] = u * r * r;
    atoms[9] = v * r * r;
    atoms[10] = u * abs_v * v;
    atoms[11] = v * abs_u * u;
    atoms[12] = r * abs_u * u;
    atoms[13] = r * abs_v * v;
    atoms[14] = u * abs_u * u;
    atoms[15] = v * abs_v * v;
    if (clamp_pif > 0.f) {
        for (float& a : atoms) {
            a = std::max(-clamp_pif, std::min(clamp_pif, a));
        }
    }
    return atoms;
}

std::vector<float> KoopmanEncoder::encode(const std::array<float, 3>& dyn_norm) const {
    if (!impl_) {
        throw std::runtime_error("KoopmanEncoder not loaded");
    }
    const std::vector<float> atoms =
        computeAtoms16(dyn_norm[0], dyn_norm[1], dyn_norm[2], impl_->clamp_pif);
    std::vector<float> h = atoms;

    for (const auto& blk : impl_->res_blocks) {
        std::vector<float> identity = blk.has_shortcut ? linearForward(blk.shortcut, h) : h;
        std::vector<float> out = linearForward(blk.fc, h);
        for (size_t i = 0; i < out.size(); ++i) {
            const float scale = blk.conv_weight[i * 3 + 1];
            const float bias = i < blk.conv_bias.size() ? blk.conv_bias[i] : 0.f;
            out[i] = detail::gelu(out[i] * scale + bias);
        }
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] += identity[i];
        }
        h = std::move(out);
    }

    const std::vector<float> hidden = linearForward(impl_->out_linear, h);
    std::vector<float> z;
    z.reserve(48);
    z.insert(z.end(), atoms.begin(), atoms.end());
    z.insert(z.end(), hidden.begin(), hidden.end());
    return z;
}

}  // namespace koopman_control
