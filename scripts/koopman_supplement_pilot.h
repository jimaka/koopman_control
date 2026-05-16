/**
 * koopman_supplement_pilot.h
 *
 * Koopman 训练集补充任务舵手，接口与 KoopmanUltraGranularPilot 一致：
 *   USVCommand cmd = pilot.update(heading_rad);
 *
 * 针对 merged 集缺口：高 u_std、非零 v/r、保持纵向速度（避免 left_turn 式 u≈0）。
 * 默认任务时长 2000 s（10 × 200 s），与 koopman_test 切段一致。
 *
 * 嵌入示例:
 *   #include "koopman_supplement_pilot.h"
 *   KoopmanSupplementPilot pilot(0.1, 0.0);
 *   while (!pilot.finished()) {
 *     USVCommand c = pilot.update(fusion_yaw);
 *     // 下发 c.port_thr, c.port_ang, c.stbd_thr, c.stbd_ang
 *   }
 */

#ifndef KOOPMAN_SUPPLEMENT_PILOT_H_
#define KOOPMAN_SUPPLEMENT_PILOT_H_

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct USVCommand {
  double port_thr;
  double port_ang;
  double stbd_thr;
  double stbd_ang;
  std::string stage;
};

/**
 * 补充辨识任务：在单一连续时间轴上排布 10 段 × 200 s 机动。
 * 油门/舵角语义与 KoopmanUltraGranularPilot 相同（油门 %，舵角 rad）。
 */
class KoopmanSupplementPilot {
 public:
  static constexpr double kDefaultMissionSec = 2000.0;
  static constexpr double kSegmentSec = 200.0;

  explicit KoopmanSupplementPilot(double dt = 0.1, double time = 0.0)
      : dt_(dt), time_(time) {
    for (int deg = 5; deg <= 35; deg += 5)
      z_amps_.push_back(deg * M_PI / 180.0);
  }

  /** 与实船控制循环一致：传入当前航向 (rad)，返回本周期推进器指令。 */
  USVCommand update(double heading_rad) {
    time_ += dt_;
    const double t = time_;

    // ==========================================
    // 【补充训练集】: 高 u_std + 前进态机动 (0 - 2000s)
    // 10 段 × 200s，对齐 convert_supplement_log_to_npz --split phase
    // ==========================================
    if (t < kDefaultMissionSec) {
      const double seg = std::floor(t / kSegmentSec);
      const double t_rel = std::fmod(t, kSegmentSec);

      switch (static_cast<int>(seg)) {
        case 0:
          return run_u_chirp_cruise(t_rel, seg == 0 ? "" : "_2");
        case 1:
          return run_high_speed_zigzag(heading_rad, t_rel, t, "SUPP_ZIGZAG_HS");
        case 2:
          return run_fig8_surge(t_rel, "SUPP_FIG8_SURGE");
        case 3:
          return run_speed_ramp(t_rel, "SUPP_SPEED_RAMP");
        case 4:
          return run_yaw_surge_combo(t_rel, "SUPP_YAW_SURGE");
        case 5:
          return run_u_chirp_cruise(t_rel, "_2");
        case 6:
          return run_high_speed_zigzag(heading_rad, t_rel, t, "SUPP_ZIGZAG_HS_2");
        case 7:
          return run_fig8_surge(t_rel, "SUPP_FIG8_SURGE_2");
        case 8:
          return run_speed_ramp(t_rel, "SUPP_SPEED_RAMP_2");
        case 9:
          return run_yaw_surge_combo(t_rel, "SUPP_YAW_SURGE_2");
        default:
          break;
      }
    }

    return {0.0, 0.0, 0.0, 0.0, "FINISHED"};
  }

  bool finished() const { return time_ >= kDefaultMissionSec; }
  double time() const { return time_; }
  double dt() const { return dt_; }
  void set_time(double t) { time_ = t; }

  /** 当前所在 200s 段索引 [0, 9]，已结束返回 -1。 */
  int segment_index() const {
    if (time_ >= kDefaultMissionSec) return -1;
    return static_cast<int>(std::floor(time_ / kSegmentSec));
  }

 private:
  double dt_;
  double time_;
  bool is_right_ = true;
  double base_heading_ = 0.0;
  std::vector<double> z_amps_;

  int zigzag_block_ = -1;
  int last_zigzag_amp_idx_ = -1;

  USVCommand run_u_chirp_cruise(double t_rel, const std::string& suffix) {
    // 油门包络 + 小幅舵角：拉高 u_std，均值油门 ~55–80%
    const double thr_base =
        55.0 + 20.0 * std::sin(2.0 * M_PI * 0.008 * t_rel)
        + 8.0 * std::sin(2.0 * M_PI * 0.05 * t_rel);
    const double thr = clamp(thr_base, 40.0, 85.0);
    const double ang = 5.0 * M_PI / 180.0 * std::sin(2.0 * M_PI * 0.03 * t_rel);
    return {thr, ang, thr, -ang, "SUPP_U_CHIRP" + suffix};
  }

  USVCommand run_high_speed_zigzag(double head, double t_rel, double t_global,
                                   const std::string& tag) {
    // 7 档舵角 × 固定大油门，与 UltraGranular TRAIN_ZIGZAG 同结构
    const double thr = 75.0;
    const int amp_idx =
        std::min(static_cast<int>(std::floor(t_rel / 28.0)),
                 static_cast<int>(z_amps_.size()) - 1);
    const double amp = z_amps_[amp_idx];

    const int block = static_cast<int>(std::floor(t_global / kSegmentSec));
    if (block != zigzag_block_ || amp_idx != last_zigzag_amp_idx_) {
      base_heading_ = head;
      is_right_ = true;
      zigzag_block_ = block;
      last_zigzag_amp_idx_ = amp_idx;
    }
    return run_zigzag(head, amp, thr, tag);
  }

  USVCommand run_fig8_surge(double t_rel, const std::string& tag) {
    // 类似 VAL_FIGURE_8，但缩短换向周期以适配 200s 段
    double steer = 22.0 * M_PI / 180.0;
    if (t_rel > 100.0) steer = -steer;
    const double thr_l = 55.0 + 12.0 * std::sin(2.0 * M_PI * 0.01 * t_rel);
    const double thr_r = 55.0 - 12.0 * std::sin(2.0 * M_PI * 0.01 * t_rel);
    return {thr_l, steer, thr_r, steer, tag};
  }

  USVCommand run_speed_ramp(double t_rel, const std::string& tag) {
    // 三角扫油门 45%–90%，每 40s 一档
    const double period = 80.0;
    const double phase01 = std::fmod(t_rel, period) / period;
    const double tri =
        (phase01 < 0.5) ? (2.0 * phase01) : (2.0 * (1.0 - phase01));
    const double thr = 45.0 + tri * 45.0;
    const double ang = 4.0 * M_PI / 180.0 * std::sin(2.0 * M_PI * 0.06 * t_rel);
    return {thr, ang, thr, -ang, tag};
  }

  USVCommand run_yaw_surge_combo(double t_rel, const std::string& tag) {
    // 差速 + 舵：保持前进（非 TRAIN_DIFF_TURN 原地转）
    const double rud = 18.0 * M_PI / 180.0 * std::sin(2.0 * M_PI * 0.02 * t_rel);
    const double thr_mid = 58.0;
    const double d_thr = 14.0 * std::sin(2.0 * M_PI * 0.02 * t_rel);
    return {thr_mid + d_thr, rud, thr_mid - d_thr, rud, tag};
  }

  USVCommand run_zigzag(double head, double amp, double thr,
                        const std::string& tag) {
    const double raw_err = head - base_heading_;
    const double wrapped_err = std::atan2(std::sin(raw_err), std::cos(raw_err));

    if (is_right_ && wrapped_err <= -amp)
      is_right_ = false;
    else if (!is_right_ && wrapped_err >= amp)
      is_right_ = true;
    const double target_ang = is_right_ ? amp : -amp;
    return {thr, target_ang, thr, target_ang, tag};
  }

  static double clamp(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
  }
};

#endif  // KOOPMAN_SUPPLEMENT_PILOT_H_
