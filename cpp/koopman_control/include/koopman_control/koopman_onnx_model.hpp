#pragma once

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

/** ONNX Runtime rollout：state0(6) + u_seq(H,4) + dt -> states(H+1,6) */
class KoopmanOnnxModel {
public:
    explicit KoopmanOnnxModel(const std::string& onnx_path);
    ~KoopmanOnnxModel();

    KoopmanOnnxModel(const KoopmanOnnxModel&) = delete;
    KoopmanOnnxModel& operator=(const KoopmanOnnxModel&) = delete;
    KoopmanOnnxModel(KoopmanOnnxModel&&) noexcept;
    KoopmanOnnxModel& operator=(KoopmanOnnxModel&&) noexcept;

    int horizon() const { return horizon_; }

    std::vector<float> rollout(const std::array<float, 6>& state0,
                               const std::vector<float>& u_seq_flat,
                               float dt) const;

private:
    int readHorizonFromSession() const;

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> options_;
    std::unique_ptr<Ort::Session> session_;
    int horizon_{0};
};

}  // namespace koopman_control
