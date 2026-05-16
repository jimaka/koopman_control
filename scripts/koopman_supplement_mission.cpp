/**
 * koopman_supplement_mission.cpp
 *
 * 离线 3-DOF 双推进器船舶仿真 + 自动航行任务序列，用于补充 Koopman 训练集缺口：
 *   - 高 u_std（纵向速度起伏，但保持 |u| > 2 m/s）
 *   - 非零 r / v（Z 字、8 字、差速，避免 left_turn 式 u≈0 原地转）
 *   - 段长 200 s @ 10 Hz，与 koopman_test / extract_left_turn 切段一致
 *
 * 输出 CSV（供 convert_supplement_log_to_npz.py 转 NPZ）：
 *   time,x,y,yaw,u,v,r,port_thr,port_angle,stbd_thr,stbd_angle,phase_id,phase_name
 *
 * 编译：
 *   g++ -O2 -std=c++17 -o koopman_supplement_mission koopman_supplement_mission.cpp
 *
 * 运行：
 *   ./koopman_supplement_mission -o supplement_mission_log.csv
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDt = 0.1;           // 10 Hz，与 split_high_density_bag.py 一致
constexpr double kSegmentSec = 200.0; // 与 test / extract_left_turn 切段一致
constexpr double kMinForwardU = 2.0;  // 补充段最低目标纵向速度 (m/s)

inline double clamp(double x, double lo, double hi) {
  return std::max(lo, std::min(hi, x));
}
inline double wrapPi(double a) {
  while (a > kPi) a -= 2.0 * kPi;
  while (a < -kPi) a += 2.0 * kPi;
  return a;
}

struct ThrusterCmd {
  double port_thr{0.0};
  double port_ang{0.0};
  double stbd_thr{0.0};
  double stbd_ang{0.0};
};

struct ShipState {
  double x{0.0}, y{0.0}, yaw{0.0};
  double u{0.0}, v{0.0}, r{0.0};
};

// 简化水平面 3-DOF + 双推进器（与实船 bag 中 Thrusters_CMD 四维一致）
struct ShipParams {
  double mass_u = 120.0;
  double mass_v = 180.0;
  double inertia_r = 420.0;
  double damp_u = 18.0;
  double damp_v = 40.0;
  double damp_r = 55.0;
  double surge_coeff = 85.0;   // 油门 → 纵向推力系数
  double sway_coeff = 32.0;
  double yaw_coeff = 28.0;
  double coriolis_uv = 25.0;
  double max_thr = 1.0;
  double max_rudder = 0.55;    // rad，约 ±31°
};

void thruster_forces(const ThrusterCmd& c, const ShipParams& p,
                     double& Fx, double& Fy, double& Mz) {
  const double tp = clamp(c.port_thr, -p.max_thr, p.max_thr);
  const double ts = clamp(c.stbd_thr, -p.max_thr, p.max_thr);
  const double ap = clamp(c.port_ang, -p.max_rudder, p.max_rudder);
  const double as = clamp(c.stbd_ang, -p.max_rudder, p.max_rudder);

  const double fpx = tp * std::cos(ap);
  const double fpy = tp * std::sin(ap);
  const double fsx = ts * std::cos(as);
  const double fsy = ts * std::sin(as);

  Fx = p.surge_coeff * (fpx + fsx);
  Fy = p.sway_coeff * (fpy + fsy);
  Mz = p.yaw_coeff * (fpy - fsy) + 0.35 * p.yaw_coeff * (tp - ts);
}

void integrate(ShipState& s, const ThrusterCmd& cmd, const ShipParams& p) {
  double Fx = 0.0, Fy = 0.0, Mz = 0.0;
  thruster_forces(cmd, p, Fx, Fy, Mz);

  const double u = s.u, v = s.v, r = s.r;
  const double du = (Fx - p.damp_u * u - 0.8 * std::abs(u) * u) / p.mass_u;
  const double dv = (Fy - p.damp_v * v + p.coriolis_uv * u * r) / p.mass_v;
  const double dr = (Mz - p.damp_r * r) / p.inertia_r;

  s.u += kDt * du;
  s.v += kDt * dv;
  s.r += kDt * dr;

  const double cy = std::cos(s.yaw), sy = std::sin(s.yaw);
  s.x += kDt * (cy * s.u - sy * s.v);
  s.y += kDt * (sy * s.u + cy * s.v);
  s.yaw = wrapPi(s.yaw + kDt * s.r);
}

enum class PhaseKind {
  UChirpCruise,      // 油门扫频 → 高 u_std，保持前进
  HighSpeedZigzag,   // 高速 Z 字
  Fig8Surge,         // 8 字机动 + 前进速度
  SpeedRamp,         // 加减速扫速
  YawSurgeCombo,     // 差速转向但 u>2
};

struct MissionPhase {
  PhaseKind kind;
  double duration_sec;
  const char* name;
};

ThrusterCmd control_for_phase(PhaseKind kind, double t_phase, double t_global) {
  ThrusterCmd c{};
  const double w = 2.0 * kPi;

  switch (kind) {
    case PhaseKind::UChirpCruise: {
      // u 目标 2.5~4.0 m/s：低频包络 + 中频抖动
      const double u_ref = 3.25 + 0.65 * std::sin(0.015 * w * t_phase)
                               + 0.35 * std::sin(0.08 * w * t_phase);
      const double base = 0.42 + 0.12 * (u_ref - 3.0);
      c.port_thr = clamp(base + 0.04 * std::sin(0.2 * w * t_phase), 0.25, 0.85);
      c.stbd_thr = clamp(base - 0.04 * std::sin(0.2 * w * t_phase), 0.25, 0.85);
      c.port_ang = 0.06 * std::sin(0.05 * w * t_phase);
      c.stbd_ang = -c.port_ang;
      break;
    }
    case PhaseKind::HighSpeedZigzag: {
      const double base = 0.58;
      const double rud = 0.38 * std::sin(0.04 * w * t_phase);
      c.port_thr = base + 0.08 * rud;
      c.stbd_thr = base - 0.08 * rud;
      c.port_ang = rud;
      c.stbd_ang = rud;
      break;
    }
    case PhaseKind::Fig8Surge: {
      // 8 字航向：较高舵角 + 左右油门差，保证 r、v 非零且 u>2
      const double yaw_rate_ref = 0.20 * std::sin(0.028 * w * t_phase);
      const double rud = clamp(2.4 * yaw_rate_ref, -0.48, 0.48);
      c.port_thr = clamp(0.52 + 0.10 * std::sin(0.056 * w * t_phase), 0.38, 0.68);
      c.stbd_thr = clamp(0.52 - 0.10 * std::sin(0.056 * w * t_phase), 0.38, 0.68);
      c.port_ang = rud;
      c.stbd_ang = rud;
      (void)t_global;
      break;
    }
    case PhaseKind::SpeedRamp: {
      // 三角扫速：2.2 → 4.0 → 2.2 m/s（约 200s 一周期内多轮）
      const double period = 80.0;
      const double phase01 = std::fmod(t_phase, period) / period;
      const double tri = (phase01 < 0.5) ? (2.0 * phase01) : (2.0 * (1.0 - phase01));
      const double u_ref = 2.2 + tri * 1.8;
      const double thr = 0.32 + 0.14 * (u_ref - 2.2);
      c.port_thr = clamp(thr, 0.28, 0.82);
      c.stbd_thr = c.port_thr;
      c.port_ang = 0.05 * std::sin(0.1 * w * t_phase);
      c.stbd_ang = -c.port_ang;
      break;
    }
    case PhaseKind::YawSurgeCombo: {
      // 差速 + 舵角：偏航明显且维持纵向
      const double rud = 0.32 * std::sin(0.035 * w * t_phase);
      c.port_thr = clamp(0.50 + 0.12 * rud, 0.38, 0.72);
      c.stbd_thr = clamp(0.50 - 0.12 * rud, 0.38, 0.72);
      c.port_ang = 0.25 * rud;
      c.stbd_ang = 0.25 * rud;
      break;
    }
  }
  return c;
}

std::vector<MissionPhase> default_mission() {
  return {
      {PhaseKind::UChirpCruise, kSegmentSec, "u_chirp_cruise"},
      {PhaseKind::HighSpeedZigzag, kSegmentSec, "high_speed_zigzag"},
      {PhaseKind::Fig8Surge, kSegmentSec, "fig8_surge"},
      {PhaseKind::SpeedRamp, kSegmentSec, "speed_ramp"},
      {PhaseKind::YawSurgeCombo, kSegmentSec, "yaw_surge_combo"},
      // 重复一轮以增加段数（共 10 段 ≈ 2000s）
      {PhaseKind::UChirpCruise, kSegmentSec, "u_chirp_cruise_2"},
      {PhaseKind::HighSpeedZigzag, kSegmentSec, "high_speed_zigzag_2"},
      {PhaseKind::Fig8Surge, kSegmentSec, "fig8_surge_2"},
      {PhaseKind::SpeedRamp, kSegmentSec, "speed_ramp_2"},
      {PhaseKind::YawSurgeCombo, kSegmentSec, "yaw_surge_combo_2"},
  };
}

void run_mission(const std::string& out_csv, bool verbose) {
  const auto mission = default_mission();
  double t_global = 0.0;
  int phase_id = 0;

  std::ofstream out(out_csv);
  if (!out) {
    std::cerr << "Cannot open output: " << out_csv << "\n";
    std::exit(1);
  }

  out << std::fixed << std::setprecision(6);
  out << "time,x,y,yaw,u,v,r,port_thr,port_angle,stbd_thr,stbd_angle,phase_id,phase_name\n";

  ShipState s{};
  ShipParams p{};
  // 初速：直接进入巡航区，避免起步段 u≈0
  s.u = 2.8;
  s.v = 0.0;
  s.r = 0.0;

  for (const auto& ph : mission) {
    const int this_phase_id = phase_id++;
    double t_phase = 0.0;
    const int steps = static_cast<int>(std::round(ph.duration_sec / kDt));

    if (verbose) {
      std::cout << ">>> Phase " << this_phase_id << " " << ph.name
                << " (" << ph.duration_sec << "s)\n";
    }

    for (int k = 0; k < steps; ++k) {
      const ThrusterCmd cmd = control_for_phase(ph.kind, t_phase, t_global);

      out << t_global << "," << s.x << "," << s.y << "," << s.yaw << ","
          << s.u << "," << s.v << "," << s.r << "," << cmd.port_thr << ","
          << cmd.port_ang << "," << cmd.stbd_thr << "," << cmd.stbd_ang << ","
          << this_phase_id << "," << ph.name << "\n";

      integrate(s, cmd, p);
      t_phase += kDt;
      t_global += kDt;
    }
  }

  std::cout << "Wrote " << out_csv << " | total_time=" << t_global
            << "s phases=" << mission.size() << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_csv = "supplement_mission_log.csv";
  bool verbose = true;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-o" || arg == "--output") {
      if (i + 1 < argc) out_csv = argv[++i];
    } else if (arg == "-q" || arg == "--quiet") {
      verbose = false;
    } else if (arg == "-h" || arg == "--help") {
      std::cout
          << "Usage: koopman_supplement_mission [-o supplement_mission_log.csv]\n"
          << "Generates 10 x 200s segments (~2000s @ 10Hz) for Koopman supplement.\n";
      return 0;
    }
  }

  run_mission(out_csv, verbose);
  return 0;
}
