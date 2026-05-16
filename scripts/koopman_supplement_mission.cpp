/**
 * koopman_supplement_mission.cpp
 *
 * 可选离线 3-DOF 仿真 + CSV 落盘；舵手逻辑在 koopman_supplement_pilot.h，
 * 与 KoopmanUltraGranularPilot 相同调用方式，便于嵌入实船程序。
 *
 * 编译:
 *   cd scripts && make
 *
 * 仅仿真落盘:
 *   ./koopman_supplement_mission -o ../supplement_mission_log.csv
 *
 * 在你自己的节点中:
 *   #include "koopman_supplement_pilot.h"
 *   KoopmanSupplementPilot pilot(0.1, 0.0);
 *   USVCommand cmd = pilot.update(fusion_yaw);
 */

#include "koopman_supplement_pilot.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

constexpr double kPi = M_PI;

inline double wrapPi(double a) {
  while (a > kPi) a -= 2.0 * kPi;
  while (a < -kPi) a += 2.0 * kPi;
  return a;
}

struct ShipState {
  double x{0.0}, y{0.0}, yaw{0.0};
  double u{2.8}, v{0.0}, r{0.0};
};

struct ShipParams {
  double mass_u = 120.0;
  double mass_v = 180.0;
  double inertia_r = 420.0;
  double damp_u = 18.0;
  double damp_v = 40.0;
  double damp_r = 55.0;
  double surge_coeff = 0.85;
  double sway_coeff = 0.32;
  double yaw_coeff = 0.28;
  double coriolis_uv = 25.0;
  double max_thr_pct = 100.0;
  double max_rudder = 35.0 * kPi / 180.0;
};

void thruster_forces(const USVCommand& c, const ShipParams& p, double& Fx,
                     double& Fy, double& Mz) {
  const auto clamp = [](double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
  };
  const double tp = clamp(c.port_thr, -p.max_thr_pct, p.max_thr_pct) / 100.0;
  const double ts = clamp(c.stbd_thr, -p.max_thr_pct, p.max_thr_pct) / 100.0;
  const double ap = clamp(c.port_ang, -p.max_rudder, p.max_rudder);
  const double as = clamp(c.stbd_ang, -p.max_rudder, p.max_rudder);

  const double fpx = tp * std::cos(ap);
  const double fpy = tp * std::sin(ap);
  const double fsx = ts * std::cos(as);
  const double fsy = ts * std::sin(as);

  Fx = 100.0 * p.surge_coeff * (fpx + fsx);
  Fy = 100.0 * p.sway_coeff * (fpy + fsy);
  Mz = 100.0 * p.yaw_coeff * (fpy - fsy) + 35.0 * p.yaw_coeff * (tp - ts);
}

void integrate(ShipState& s, const USVCommand& cmd, const ShipParams& p,
               double dt) {
  double Fx = 0.0, Fy = 0.0, Mz = 0.0;
  thruster_forces(cmd, p, Fx, Fy, Mz);

  const double u = s.u, v = s.v, r = s.r;
  s.u += dt * (Fx - p.damp_u * u - 0.8 * std::abs(u) * u) / p.mass_u;
  s.v += dt * (Fy - p.damp_v * v + p.coriolis_uv * u * r) / p.mass_v;
  s.r += dt * (Mz - p.damp_r * r) / p.inertia_r;

  const double cy = std::cos(s.yaw), sy = std::sin(s.yaw);
  s.x += dt * (cy * u - sy * v);
  s.y += dt * (sy * u + cy * v);
  s.yaw = wrapPi(s.yaw + dt * s.r);
}

void run_offline_log(const std::string& out_csv, double dt, bool verbose) {
  KoopmanSupplementPilot pilot(dt, 0.0);
  pilot.set_time(0.0);

  std::ofstream out(out_csv);
  if (!out) {
    std::cerr << "Cannot open output: " << out_csv << "\n";
    std::exit(1);
  }

  out << std::fixed << std::setprecision(6);
  out << "time,x,y,yaw,u,v,r,port_thr,port_angle,stbd_thr,stbd_angle,phase_id,"
         "phase_name\n";

  ShipState s{};
  ShipParams params{};
  int logged_phase = -1;

  while (!pilot.finished()) {
    const double t_log = pilot.time();
    const int phase_id = pilot.segment_index();
    USVCommand cmd = pilot.update(s.yaw);

    out << t_log << "," << s.x << "," << s.y << "," << s.yaw << "," << s.u
        << "," << s.v << "," << s.r << "," << cmd.port_thr << ","
        << cmd.port_ang << "," << cmd.stbd_thr << "," << cmd.stbd_ang << ","
        << phase_id << "," << cmd.stage << "\n";

    if (verbose && phase_id != logged_phase) {
      logged_phase = phase_id;
      std::cout << ">>> segment " << phase_id << " " << cmd.stage << "\n";
    }

    integrate(s, cmd, params, dt);
  }

  std::cout << "Wrote " << out_csv << " | duration="
            << KoopmanSupplementPilot::kDefaultMissionSec << "s @ " << dt
            << "Hz\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_csv = "supplement_mission_log.csv";
  double dt = 0.1;
  bool verbose = true;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-o" || arg == "--output") {
      if (i + 1 < argc) out_csv = argv[++i];
    } else if (arg == "--dt") {
      if (i + 1 < argc) dt = std::atof(argv[++i]);
    } else if (arg == "-q" || arg == "--quiet") {
      verbose = false;
    } else if (arg == "-h" || arg == "--help") {
      std::cout
          << "Usage: koopman_supplement_mission [-o csv] [--dt 0.1]\n"
          << "Pilot header: koopman_supplement_pilot.h (embed in your stack)\n"
          << "Mission: " << KoopmanSupplementPilot::kDefaultMissionSec
          << "s = 10 x " << KoopmanSupplementPilot::kSegmentSec << "s segments\n";
      return 0;
    }
  }

  run_offline_log(out_csv, dt, verbose);
  return 0;
}
