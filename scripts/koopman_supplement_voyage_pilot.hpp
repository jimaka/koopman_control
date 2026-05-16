/**
 * koopman_supplement_voyage_pilot.hpp
 *
 * 【后加入】Koopman 数据集补充跑船流程，与 KoopmanUltraGranularPilot 接口对齐、类名不同。
 * 在主编队任务（0–25200s）之外单独调用，用于补高 u_std / 非零 v·r 段。
 *
 * 用法（与 UltraGranular 相同）:
 *   #include "koopman_supplement_voyage_pilot.hpp"
 *   KoopmanSupplementVoyagePilot pilot(0.1, 0.0);
 *   while (running) {
 *     USVCommand cmd = pilot.update(heading_rad);
 *     // 下发 cmd.port_thr, cmd.port_ang, cmd.stbd_thr, cmd.stbd_ang
 *   }
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#ifndef KOOPMAN_USV_COMMAND_DEFINED
#define KOOPMAN_USV_COMMAND_DEFINED
struct USVCommand {
  double port_thr;
  double port_ang;
  double stbd_thr;
  double stbd_ang;
  std::string stage;
};
#endif

/**
 * 补充航行舵手：2000 s（10×200 s），油门 %、舵角 rad。
 * 对应 convert_supplement_log_to_npz.py --split phase。
 */
class KoopmanSupplementVoyagePilot {
 public:
  KoopmanSupplementVoyagePilot(double dt = 0.1, double time = 0.0)
      : dt_(dt), time_(time) {
    for (int deg = 5; deg <= 35; deg += 5)
      z_amps_.push_back(deg * M_PI / 180.0);
  }

  USVCommand update(double heading_rad) {
    time_ += dt_;

    // ==========================================
    // 【补充训练集】: 高密度缺口填补 (0 - 2000s)
    // ==========================================
    if (time_ < 2000.0) {
      const int seg = static_cast<int>(std::floor(time_ / 200.0));
      const double t_rel = std::fmod(time_, 200.0);

      switch (seg) {
        case 0:
          return exec_u_chirp_cruise(t_rel, "");
        case 1:
          return exec_high_speed_zigzag(heading_rad, t_rel, time_,
                                        "SUPP_ZIGZAG_HS");
        case 2:
          return exec_fig8_surge(t_rel, "SUPP_FIG8_SURGE");
        case 3:
          return exec_speed_ramp(t_rel, "SUPP_SPEED_RAMP");
        case 4:
          return exec_yaw_surge_combo(t_rel, "SUPP_YAW_SURGE");
        case 5:
          return exec_u_chirp_cruise(t_rel, "_2");
        case 6:
          return exec_high_speed_zigzag(heading_rad, t_rel, time_,
                                        "SUPP_ZIGZAG_HS_2");
        case 7:
          return exec_fig8_surge(t_rel, "SUPP_FIG8_SURGE_2");
        case 8:
          return exec_speed_ramp(t_rel, "SUPP_SPEED_RAMP_2");
        case 9:
          return exec_yaw_surge_combo(t_rel, "SUPP_YAW_SURGE_2");
        default:
          break;
      }
    }

    return {0.0, 0.0, 0.0, 0.0, "FINISHED"};
  }

  bool is_finished() const { return time_ >= 2000.0; }

 private:
  double dt_, time_;
  bool is_right = true;
  double base_heading_ = 0.0;
  std::vector<double> z_amps_;
  int last_amp_idx_ = -1;
  int last_seg_idx_ = -1;

  USVCommand exec_u_chirp_cruise(double t_rel, const std::string& suffix) {
    const double thr = clamp_val(
        55.0 + 20.0 * std::sin(2.0 * M_PI * 0.008 * t_rel)
            + 8.0 * std::sin(2.0 * M_PI * 0.05 * t_rel),
        40.0, 85.0);
    const double ang = 5.0 * M_PI / 180.0 * std::sin(2.0 * M_PI * 0.03 * t_rel);
    return {thr, ang, thr, -ang, "SUPP_U_CHIRP" + suffix};
  }

  USVCommand exec_high_speed_zigzag(double head, double t_rel, double t_global,
                                    const std::string& tag) {
    const double thr = 75.0;
    int amp_idx = static_cast<int>(std::floor(t_rel / 28.0));
    amp_idx = std::min(amp_idx, static_cast<int>(z_amps_.size()) - 1);
    const double amp = z_amps_[amp_idx];

    const int seg_idx = static_cast<int>(std::floor(t_global / 200.0));
    if (seg_idx != last_seg_idx_ || amp_idx != last_amp_idx_) {
      base_heading_ = head;
      is_right = true;
      last_seg_idx_ = seg_idx;
      last_amp_idx_ = amp_idx;
    }
    return run_zigzag(head, amp, thr, tag);
  }

  USVCommand exec_fig8_surge(double t_rel, const std::string& tag) {
    double steer = 22.0 * M_PI / 180.0;
    if (t_rel > 100.0) steer = -steer;
    const double thr_l = 55.0 + 12.0 * std::sin(2.0 * M_PI * 0.01 * t_rel);
    const double thr_r = 55.0 - 12.0 * std::sin(2.0 * M_PI * 0.01 * t_rel);
    return {thr_l, steer, thr_r, steer, tag};
  }

  USVCommand exec_speed_ramp(double t_rel, const std::string& tag) {
    const double period = 80.0;
    const double phase01 = std::fmod(t_rel, period) / period;
    const double tri =
        (phase01 < 0.5) ? (2.0 * phase01) : (2.0 * (1.0 - phase01));
    const double thr = 45.0 + tri * 45.0;
    const double ang = 4.0 * M_PI / 180.0 * std::sin(2.0 * M_PI * 0.06 * t_rel);
    return {thr, ang, thr, -ang, tag};
  }

  USVCommand exec_yaw_surge_combo(double t_rel, const std::string& tag) {
    const double rud = 18.0 * M_PI / 180.0 * std::sin(2.0 * M_PI * 0.02 * t_rel);
    const double thr_mid = 58.0;
    const double d_thr = 14.0 * std::sin(2.0 * M_PI * 0.02 * t_rel);
    return {thr_mid + d_thr, rud, thr_mid - d_thr, rud, tag};
  }

  USVCommand run_zigzag(double head, double amp, double thr,
                        const std::string& tag) {
    double raw_err = head - base_heading_;
    double wrapped_err = std::atan2(std::sin(raw_err), std::cos(raw_err));

    if (is_right && wrapped_err <= -amp)
      is_right = false;
    else if (!is_right && wrapped_err >= amp)
      is_right = true;
    double target_ang = is_right ? amp : -amp;
    return {thr, target_ang, thr, target_ang, tag};
  }

  static double clamp_val(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
  }
};
