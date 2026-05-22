/** 对照 Python rollout_check.npz 验证 ONNX C++ 前向 */
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "koopman_control/koopman_onnx_model.hpp"

namespace {

bool loadNpzRollout(const std::string& path,
                    std::array<float, 6>& state0,
                    std::vector<float>& u_flat,
                    int& H,
                    std::vector<float>& states_ref) {
    const std::string txt = path + ".txt";
    std::ifstream in(txt);
    if (!in) {
        return false;
    }
    for (int i = 0; i < 6; ++i) {
        in >> state0[i];
    }
    in >> H;
    const int nu = H * 4;
    u_flat.resize(nu);
    for (int i = 0; i < nu; ++i) {
        in >> u_flat[i];
    }
    int rows = 0;
    int cols = 0;
    in >> rows >> cols;
    states_ref.resize(rows * cols);
    for (size_t i = 0; i < states_ref.size(); ++i) {
        in >> states_ref[i];
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string onnx = "cpp/koopman_mpc/weights/koopman_rollout.onnx";
    std::string check_txt = "cpp/koopman_mpc/weights/rollout_check.npz";
    if (argc > 1) {
        onnx = argv[1];
    }
    if (argc > 2) {
        check_txt = argv[2];
    }

    std::array<float, 6> s0{};
    std::vector<float> u_flat;
    std::vector<float> states_ref;
    int H = 0;
    if (!loadNpzRollout(check_txt, s0, u_flat, H, states_ref)) {
        std::cerr << "Missing " << check_txt << ".txt — run write_rollout_check_txt.py first\n";
        return 1;
    }

    try {
        koopman_control::KoopmanOnnxModel model(onnx);
        auto states = model.rollout(s0, u_flat, 1.0f);
        const int rows = static_cast<int>(states.size() / 6);
        float max_err = 0.f;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < 6; ++j) {
                const float e = std::fabs(states[i * 6 + j] - states_ref[i * 6 + j]);
                max_err = std::max(max_err, e);
            }
        }
        std::cout << "rollout max_abs_err vs Python: " << max_err << "\n";
        if (max_err > 1e-3f) {
            std::cerr << "FAIL\n";
            return 1;
        }
        std::cout << "rollout verify OK\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
