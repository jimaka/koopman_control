/**
 * @file motion_integration_example.cpp
 * @brief motion.cpp 集成示例（无 ROS 依赖）
 *
 * 演示：加载潜空间 YAML → 构造参考点 → 调用 KoopmanMotionMpc::solve（OSQP，无需 ONNX）
 *
 * 编译示例（在 koopman_control/build 目录）：
 *   g++ -std=c++17 ../examples/motion_integration_example.cpp \
 *       -I../include -L. -lkoopman_control \
 *       -lyaml-cpp -losqpstatic
 */
#include <iostream>
#include <vector>

#include "koopman_control/motion_bridge.hpp"
#include "koopman_control/mpc_config_loader.hpp"

int main() {
    try {
        const std::string yaml_path = "cpp/koopman_control/config/mpc_config.yaml";

        // 1. 加载 MPC 参数（含 latent_model 路径）
        koopman_control::MpcConfig cfg =
            koopman_control::loadMpcConfigFromYaml(yaml_path);

        // 2. 配置 motion 参考时间轴（与 PointChange 中 mpc_during 对齐）
        koopman_control::MotionBridgeConfig bridge;
        bridge.ref_dt = 1.0f;
        bridge.ref_time_offset = 0.5f;

        // 实船 MPC 仅需潜空间 YAML（OSQP 求解，无需 ONNX）
        koopman_control::KoopmanMotionMpc mpc(cfg.latent_model, cfg, bridge);
        std::cout << "MPC horizon = " << mpc.horizon() << "\n";

        // 3. 构造输入：模拟 PointChange 生成的 40 个船体坐标参考点
        koopman_control::MotionSolveInput in;
        in.u = 1.5f;
        in.v = 0.05f;
        in.r = 0.01f;
        in.ref.resize(40);
        for (size_t i = 0; i < in.ref.size(); ++i) {
            in.ref[i].x = static_cast<float>(i) * 0.5f;
            in.ref[i].y = 0.f;
            in.ref[i].psi = 0.f;
            in.ref[i].u = 1.5f;
            in.ref[i].v = 0.f;
            in.ref[i].r = 0.f;
        }

        // 4. 单步 MPC 求解
        koopman_control::MotionSolveOutput out;
        if (!mpc.solve(in, out)) {
            std::cerr << "solve failed\n";
            return 1;
        }

        std::cout << "cost=" << out.cost << " u=[";
        for (float c : out.control) {
            std::cout << c << " ";
        }
        std::cout << "]\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
