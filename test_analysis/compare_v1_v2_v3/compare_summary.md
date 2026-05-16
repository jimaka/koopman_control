# Comparison summary

| tag | ckpt | n_samples | pred_len | agg.vel_rmse_mean | agg.vel_rmse_step_1 | agg.vel_rmse_step_5 | agg.vel_rmse_step_10 | agg.vel_rmse_step_20 | agg.u_rmse_step_20 | agg.v_rmse_step_20 | agg.r_rmse_step_20 | agg.acc_rmse_mean | agg.traj_xy_rmse_step_20 | div.ratio_step20_over_step1 | div.slope_loglog | div.slope_linear | div.lyapunov_like | div.auc_error_curve | div.monotonic_increasing | div.instability_score | div.divergent_sample_pct | bias.u_bias_mean | bias.v_bias_mean | bias.r_bias_mean | tail.vel_err_step20_p50 | tail.vel_err_step20_p90 | tail.vel_err_step20_p99 | tail.vel_err_step20_max | fit.u_r2_mean | fit.v_r2_mean | fit.r_r2_mean | fit.u_corr_mean | fit.v_corr_mean | fit.r_corr_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | checkpoints/koopman_v1_best.pth | 35622 | 20 | 0.0135935 | 0.00131387 | 0.00654884 | 0.0130207 | 0.0256436 | 0.0225416 | 0.0122256 | 0.000749803 | 0.0129368 | 0.0277499 | 19.5176 | 0.991456 | 0.00128118 | 0.156385 | 0.0129196 | True | 2.15294 | 99.8793 | -0.00365191 | 8.74499e-05 | 2.73778e-05 | 0.0167561 | 0.0362113 | 0.0836092 | 0.162472 | 0.999007 | 0.999324 | 0.999526 | 0.999551 | 0.999829 | 0.999886 |
| v2 | checkpoints/koopman_v2_best.pth | 35622 | 20 | 0.010369 | 0.00184738 | 0.00728396 | 0.0109818 | 0.016059 | 0.0158045 | 0.00284718 | 0.000653175 | 0.010454 | 0.020081 | 8.69286 | 0.669505 | 0.000657704 | 0.113816 | 0.00992136 | True | 1.23633 | 81.7753 | -0.00255312 | -0.000351623 | -1.63898e-05 | 0.00937762 | 0.0251606 | 0.0495588 | 0.0949102 | 0.99935 | 0.999965 | 0.999663 | 0.999738 | 0.999987 | 0.999914 |
| v3 | checkpoints/koopman_v3_best.pth | 35622 | 20 | 0.00702127 | 0.00121624 | 0.00455221 | 0.00724467 | 0.0110681 | 0.0101008 | 0.00452493 | 0.000755289 | 0.00616204 | 0.0141993 | 9.10022 | 0.711627 | 0.000488671 | 0.116226 | 0.00671416 | True | 1.28117 | 90.5255 | 0.000378947 | 0.000154354 | 4.75032e-05 | 0.00914726 | 0.0173452 | 0.0229841 | 0.0315788 | 0.999723 | 0.999923 | 0.999576 | 0.999864 | 0.999975 | 0.999896 |

## Verdict

- **vel_rmse_step_20 最低**: `v3` (0.0110681 m/s)
- **slope_loglog 最低（最不发散）**: `v2` (0.669505)
- **instability_score 最低**: `v2` (1.23633)
- ⚠️ `instability_score` 未随版本顺序单调下降，请检查改进方向。

### 9.4 硬性指标门槛（first → last，正号 = 改善 / 下降）
- `vel_rmse_step_20`: 0.0256436 → 0.0110681  (↓ +56.8%)
- `slope_loglog`: 0.991456 → 0.711627  (↓ +28.2%)
- `instability_score`: 2.15294 → 1.28117  (↓ +40.5%)
- 条件 A (vel↓≥30% **且** slope↓≥20%) = **True**
- 条件 B (inst↓≥25% **且** vel 不上升) = **True**
- **✅ PASS** — 9.4 验收。

### §7 12 条阈值逐条判定 (test set, K=20)

| # | 指标 | v3 实测 | 阈值 | 判定 |
|---|---|---|---|---|
| G1 | vel_rmse_step_K | 0.0110681 | ≤ 0.01285 | ✅ |
| G2 | u_rmse_step_K | 0.0101008 | ≤ 0.01185 | ✅ |
| G3 | |u_bias_mean| | 0.000378947 | ≤ 0.00153 | ✅ |
| G4 | slope_loglog | 0.711627 | ≤ 0.6695 | ❌ |
| G5 | instability_score | 1.28117 | ≤ 1.2363 | ❌ |
| S1 | per_segment.worst_vel_rmse_K | 0.0141883 | ≤ 0.01569 | ✅ |
| S2 | per_segment.ratio_worst_over_best | 2.21534 | ≤ 2.78 | ✅ |
| S3 | per_segment.high_speed_seg_mean | 0.00867062 | ≤ 0.01144 | ✅ |
| S4 | degraded_pct (vs v1) | 22.9381 | ≤ 20 | ❌ |
| N1 | vel_rmse_step_1 | 0.00121624 | ≤ 0.0024 | ✅ |
| N2 | traj_xy_rmse_step_K | 0.0141993 | ≤ 0.02108 | ✅ |
| N3 | spectral_radius (I+A) | 0.997886 | ≤ 1.005 | ✅ |

**FAIL** — §7 v3 验收。

### 3 阶字典效果归因（v3 vs v2，全部数字来自 summary.json）

| 问题 | v2 | v3 | v3/v2 | 判定 |
|---|---|---|---|---|
| u_bias 漂移斜率 |slope| | 0.000374096 | 7.84012e-05 | 0.209575 | ✅ (<0.6) |
| worst-seg vel_rmse@K | 0.0196107 | 0.0141883 | 0.723499 | ✅ (<0.80) |
| (high/v2)/(low/v2) 高速相对低速改善 | 0.0141173 / 0.0105635 | 0.00867062 / 0.0107685 | 0.602494 | ✅ (<0.95) |
