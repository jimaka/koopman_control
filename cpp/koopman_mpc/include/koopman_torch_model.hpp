#pragma once

#include <string>
#include <torch/script.h>
#include <torch/torch.h>

namespace koopman_mpc {

/** 加载 TorchScript rollout：state0(6) + u_seq(H,4) -> states(H+1,6)

TorchScript 在导出时固定 ``traced_horizon``（默认 20），``u_seq`` 行数必须与之相同。
*/
class KoopmanTorchModel {
public:
    static constexpr int kTracedHorizon = 20;

    explicit KoopmanTorchModel(const std::string& ts_path);

    torch::Tensor rollout(const torch::Tensor& state0,
                          const torch::Tensor& u_seq,
                          float dt) const;

private:
    mutable torch::jit::script::Module module_;
};

}  // namespace koopman_mpc
