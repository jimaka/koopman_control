#include "koopman_onnx_model.hpp"

#include <onnxruntime_cxx_api.h>

#include <stdexcept>

namespace koopman_mpc {

KoopmanOnnxModel::KoopmanOnnxModel(const std::string& onnx_path)
    : env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "koopman_mpc")),
      options_(std::make_unique<Ort::SessionOptions>()) {
    options_->SetIntraOpNumThreads(1);
    options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    try {
        session_ = std::make_unique<Ort::Session>(*env_, onnx_path.c_str(), *options_);
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("ONNX load failed: ") + e.what());
    }
}

KoopmanOnnxModel::~KoopmanOnnxModel() = default;

KoopmanOnnxModel::KoopmanOnnxModel(KoopmanOnnxModel&&) noexcept = default;
KoopmanOnnxModel& KoopmanOnnxModel::operator=(KoopmanOnnxModel&&) noexcept = default;

std::vector<float> KoopmanOnnxModel::rollout(const std::array<float, 6>& state0,
                                             const std::vector<float>& u_seq_flat,
                                             float dt) const {
    const int64_t H = kTracedHorizon;
    if (static_cast<int64_t>(u_seq_flat.size()) != H * 4) {
        throw std::runtime_error("u_seq_flat size must be H*4");
    }

    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> s0_shape{6};
    std::vector<int64_t> u_shape{H, 4};
    std::vector<int64_t> dt_shape{};

    Ort::Value s0_tensor = Ort::Value::CreateTensor<float>(
        mem_info, const_cast<float*>(state0.data()), 6, s0_shape.data(), s0_shape.size());
    Ort::Value u_tensor = Ort::Value::CreateTensor<float>(
        mem_info, const_cast<float*>(u_seq_flat.data()), u_seq_flat.size(), u_shape.data(),
        u_shape.size());
    Ort::Value dt_tensor =
        Ort::Value::CreateTensor<float>(mem_info, &dt, 1, dt_shape.data(), dt_shape.size());

    const char* input_names[] = {"state0", "u_seq", "dt"};
    const char* output_names[] = {"states"};
    std::array<Ort::Value, 3> inputs{std::move(s0_tensor), std::move(u_tensor), std::move(dt_tensor)};

    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(),
                                 output_names, 1);

    float* out_data = outputs[0].GetTensorMutableData<float>();
    auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
    const size_t n = out_info.GetElementCount();
    return std::vector<float>(out_data, out_data + n);
}

}  // namespace koopman_mpc
