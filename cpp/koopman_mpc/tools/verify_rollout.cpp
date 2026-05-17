/** 对照 Python rollout_check.npz 验证 TorchScript C++ 前向 */
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "koopman_torch_model.hpp"

namespace {

bool loadNpzRollout(const std::string& path,
                    std::vector<float>& state0,
                    std::vector<float>& u_flat,
                    int& H,
                    std::vector<float>& states_ref) {
    // 极简：从 Python 导出的 .txt 侧车文件读取（由 verify 脚本写入）
    const std::string txt = path + ".txt";
    std::ifstream in(txt);
    if (!in) {
        return false;
    }
    int n_state0 = 6;
    state0.resize(6);
    for (int i = 0; i < n_state0; ++i) {
        in >> state0[i];
    }
    in >> H;
    int nu = H * 4;
    u_flat.resize(nu);
    for (int i = 0; i < nu; ++i) {
        in >> u_flat[i];
    }
    int rows, cols;
    in >> rows >> cols;
    states_ref.resize(rows * cols);
    for (size_t i = 0; i < states_ref.size(); ++i) {
        in >> states_ref[i];
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string ts = "cpp/koopman_mpc/weights/koopman_rollout.pt";
    std::string check_txt = "cpp/koopman_mpc/weights/rollout_check.npz.txt";
    if (argc > 1) {
        ts = argv[1];
    }
    if (argc > 2) {
        check_txt = argv[2];
    }

    std::vector<float> s0, u_flat, states_ref;
    int H = 0;
    if (!loadNpzRollout(check_txt, s0, u_flat, H, states_ref)) {
        std::cerr << "Missing " << check_txt << " — run verify_cpp.py first\n";
        return 1;
    }

    try {
        koopman_mpc::KoopmanTorchModel model(ts);
        auto state0 = torch::from_blob(s0.data(), {6}, torch::kFloat32).clone();
        auto u_seq = torch::from_blob(u_flat.data(), {H, 4}, torch::kFloat32).clone();
        auto states = model.rollout(state0, u_seq, 0.1f);
        auto acc = states.accessor<float, 2>();
        float max_err = 0.f;
        for (int i = 0; i < states.size(0); ++i) {
            for (int j = 0; j < 6; ++j) {
                const float e = std::fabs(acc[i][j] - states_ref[i * 6 + j]);
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
