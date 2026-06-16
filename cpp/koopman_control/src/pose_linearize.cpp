/**
 * @file pose_linearize.cpp
 */

#include "koopman_control/pose_linearize.hpp"

#include <cmath>

#include "koopman_control/koopman_decoder.hpp"
#include "koopman_control/koopman_latent_model.hpp"

namespace koopman_control {

PoseLinearization buildPoseLinearization(const KoopmanLatentModel& model,
                                         const KoopmanDecoder& decoder,
                                         const std::vector<float>& z0,
                                         const std::array<float, 3>& pose0,
                                         const std::vector<float>& u_tilde_U0,
                                         const std::vector<float>& pose_ref,
                                         float dt, float w_xy, float w_yaw) {
    PoseLinearization out;
    const int N = model.horizon();
    const int nz = model.nz();
    const int nu = model.nu();
    const int nvar = nu * N;
    if (!decoder.loaded() || static_cast<int>(u_tilde_U0.size()) != nvar ||
        static_cast<int>(pose_ref.size()) != 3 * N) {
        out.valid = false;
        return out;
    }

    // 标称潜变量轨迹 z_1..z_N（与 Theta 一致的精确线性传播）
    const std::vector<float> Z0 = model.predictStacked(z0, u_tilde_U0);
    const detail::Matrix& Theta = model.Theta();

    // 逐步解码物理速度 d_m (m=1..N) 与 Jacobian Jp_m (3 x nz)，并构造 V_m (3 x nvar)。
    std::vector<std::array<float, 3>> d(static_cast<size_t>(N + 1));
    std::vector<detail::Matrix> V(static_cast<size_t>(N + 1));  // V[m] 对 m=1..N 有效
    std::vector<float> zk(static_cast<size_t>(nz));
    for (int m = 1; m <= N; ++m) {
        for (int i = 0; i < nz; ++i) {
            zk[static_cast<size_t>(i)] = Z0[static_cast<size_t>((m - 1) * nz + i)];
        }
        d[static_cast<size_t>(m)] = decoder.decodePhysical(zk);
        const detail::Matrix Jp = decoder.jacobianPhysical(zk);  // 3 x nz

        // Theta 行块 (m-1)*nz .. m*nz  →  Theta_block (nz x nvar)
        // V_m = Jp (3 x nz) · Theta_block (nz x nvar)
        detail::Matrix Vm(3, nvar, 0.f);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < nvar; ++c) {
                float s = 0.f;
                for (int j = 0; j < nz; ++j) {
                    s += Jp(r, j) * Theta((m - 1) * nz + j, c);
                }
                Vm(r, c) = s;
            }
        }
        V[static_cast<size_t>(m)] = std::move(Vm);
    }

    // 标称位姿 p0_0..p0_N（船体系欧拉积分，沿用 rollout 约定：vel_m 配 yaw_{m-1}）
    std::vector<std::array<float, 3>> p0(static_cast<size_t>(N + 1));
    p0[0] = pose0;
    for (int m = 1; m <= N; ++m) {
        const float yaw_prev = p0[static_cast<size_t>(m - 1)][2];
        const float um = d[static_cast<size_t>(m)][0];
        const float vm = d[static_cast<size_t>(m)][1];
        const float rm = d[static_cast<size_t>(m)][2];
        const float c = std::cos(yaw_prev);
        const float s = std::sin(yaw_prev);
        p0[static_cast<size_t>(m)][0] = p0[static_cast<size_t>(m - 1)][0] + (um * c - vm * s) * dt;
        p0[static_cast<size_t>(m)][1] = p0[static_cast<size_t>(m - 1)][1] + (um * s + vm * c) * dt;
        p0[static_cast<size_t>(m)][2] = yaw_prev + rm * dt;
    }

    // 灵敏度递推：S* 为 p_m 对 U 的偏导（nvar 维行向量）
    detail::Matrix Phi(3 * N, nvar, 0.f);
    std::vector<float> Sx(static_cast<size_t>(nvar), 0.f);
    std::vector<float> Sy(static_cast<size_t>(nvar), 0.f);
    std::vector<float> Spsi(static_cast<size_t>(nvar), 0.f);
    for (int m = 1; m <= N; ++m) {
        const float yaw_prev = p0[static_cast<size_t>(m - 1)][2];
        const float um = d[static_cast<size_t>(m)][0];
        const float vm = d[static_cast<size_t>(m)][1];
        const float c = std::cos(yaw_prev);
        const float s = std::sin(yaw_prev);
        const float dxdpsi = (-um * s - vm * c) * dt;  // ∂x_m/∂yaw_{m-1}
        const float dydpsi = (um * c - vm * s) * dt;   // ∂y_m/∂yaw_{m-1}
        const detail::Matrix& Vm = V[static_cast<size_t>(m)];

        std::vector<float> nSx(static_cast<size_t>(nvar));
        std::vector<float> nSy(static_cast<size_t>(nvar));
        std::vector<float> nSpsi(static_cast<size_t>(nvar));
        for (int j = 0; j < nvar; ++j) {
            const float Vu = Vm(0, j);
            const float Vv = Vm(1, j);
            const float Vr = Vm(2, j);
            nSx[static_cast<size_t>(j)] =
                Sx[static_cast<size_t>(j)] + dt * (c * Vu - s * Vv) + dxdpsi * Spsi[static_cast<size_t>(j)];
            nSy[static_cast<size_t>(j)] =
                Sy[static_cast<size_t>(j)] + dt * (s * Vu + c * Vv) + dydpsi * Spsi[static_cast<size_t>(j)];
            nSpsi[static_cast<size_t>(j)] = Spsi[static_cast<size_t>(j)] + dt * Vr;
        }
        const int rx = (m - 1) * 3;
        for (int j = 0; j < nvar; ++j) {
            Phi(rx + 0, j) = nSx[static_cast<size_t>(j)];
            Phi(rx + 1, j) = nSy[static_cast<size_t>(j)];
            Phi(rx + 2, j) = nSpsi[static_cast<size_t>(j)];
        }
        Sx = std::move(nSx);
        Sy = std::move(nSy);
        Spsi = std::move(nSpsi);
    }

    // 偏置 b = nominal_err − Phi·U0（yaw 分量 wrap）
    const std::vector<float> PhiU0 = detail::Matrix::matvec(Phi, u_tilde_U0);
    out.b.assign(static_cast<size_t>(3 * N), 0.f);
    out.wq.assign(static_cast<size_t>(3 * N), 0.f);
    for (int m = 1; m <= N; ++m) {
        const int r = (m - 1) * 3;
        const float ex = p0[static_cast<size_t>(m)][0] - pose_ref[static_cast<size_t>(r + 0)];
        const float ey = p0[static_cast<size_t>(m)][1] - pose_ref[static_cast<size_t>(r + 1)];
        const float epsi = detail::wrapAngle(p0[static_cast<size_t>(m)][2] - pose_ref[static_cast<size_t>(r + 2)]);
        out.b[static_cast<size_t>(r + 0)] = ex - PhiU0[static_cast<size_t>(r + 0)];
        out.b[static_cast<size_t>(r + 1)] = ey - PhiU0[static_cast<size_t>(r + 1)];
        out.b[static_cast<size_t>(r + 2)] = epsi - PhiU0[static_cast<size_t>(r + 2)];
        out.wq[static_cast<size_t>(r + 0)] = w_xy;
        out.wq[static_cast<size_t>(r + 1)] = w_xy;
        out.wq[static_cast<size_t>(r + 2)] = w_yaw;
    }

    out.Phi = std::move(Phi);
    out.valid = true;
    return out;
}

}  // namespace koopman_control
