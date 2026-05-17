#include "koopman_torch_model.hpp"

#include <optional>
#include <stdexcept>

namespace koopman_mpc {

KoopmanTorchModel::KoopmanTorchModel(const std::string& ts_path) {
    try {
        module_ = torch::jit::load(ts_path);
    } catch (const c10::Error& e) {
        throw std::runtime_error(std::string("TorchScript load failed: ") + e.what());
    }
    module_.eval();
}

torch::Tensor KoopmanTorchModel::rollout(const torch::Tensor& state0,
                                         const torch::Tensor& u_seq,
                                         float dt) const {
    if (u_seq.size(0) != kTracedHorizon) {
        throw std::runtime_error(
            "u_seq horizon must match traced model (" + std::to_string(kTracedHorizon) +
            "), got " + std::to_string(u_seq.size(0)));
    }
    std::optional<torch::NoGradGuard> no_grad;
    if (!torch::GradMode::is_enabled()) {
        no_grad.emplace();
    }
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(state0);
    inputs.push_back(u_seq);
    inputs.push_back(torch::tensor(dt, torch::dtype(torch::kFloat32)));
    auto out = module_.forward(inputs).toTensor();
    return out;
}

}  // namespace koopman_mpc
