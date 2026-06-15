/** 验证 C++ Gamma/Theta 与 Python 参考一致，并测试 encode。 */
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "koopman_control/koopman_encode.hpp"
#include "koopman_control/koopman_latent_model.hpp"

namespace {

bool loadVec(const std::string& path, std::vector<float>& out) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    int n = 0;
    in >> n;
    out.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        in >> out[static_cast<size_t>(i)];
    }
    return true;
}

float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return 1e9f;
    }
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    std::string yaml = "cpp/koopman_mpc/weights/koopman_v4_latent.yaml";
    std::string ref_dir = "eval_out/latent_qp_cpp_ref";
    int horizon = 20;
    if (argc > 1) {
        yaml = argv[1];
    }
    if (argc > 2) {
        ref_dir = argv[2];
    }
    if (argc > 3) {
        horizon = std::stoi(argv[3]);
    }

    koopman_control::KoopmanLatentModel model;
    model.loadFromYaml(yaml, horizon);
    model.precomputePredictionMatrices();

    koopman_control::KoopmanEncoder enc;
    enc.loadFromYaml(yaml);

    std::vector<float> z0, U, Z_ref;
    if (!loadVec(ref_dir + "/z0.txt", z0) || !loadVec(ref_dir + "/U.txt", U) ||
        !loadVec(ref_dir + "/Z_ref.txt", Z_ref)) {
        std::cerr << "Missing ref vectors in " << ref_dir << " — run tests/export_latent_qp_cpp_ref.py\n";
        return 1;
    }

    const auto Z_cpp = model.predictStacked(z0, U);
    const float pred_err = maxAbsDiff(Z_cpp, Z_ref);
    std::cout << "predictStacked max_abs_err=" << pred_err << "\n";

    std::vector<float> dyn_n;
    if (loadVec(ref_dir + "/dyn_norm.txt", dyn_n) && dyn_n.size() == 3) {
        const auto z_enc = enc.encode({dyn_n[0], dyn_n[1], dyn_n[2]});
        std::vector<float> z_enc_ref;
        if (loadVec(ref_dir + "/z_encode_ref.txt", z_enc_ref)) {
            const float enc_err = maxAbsDiff(z_enc, z_enc_ref);
            std::cout << "encode max_abs_err=" << enc_err << "\n";
            if (enc_err > 1e-4f) {
                return 2;
            }
        }
    }

    if (pred_err > 1e-3f) {
        return 3;
    }
    std::cout << "[OK] latent QP matrices verified\n";
    return 0;
}
