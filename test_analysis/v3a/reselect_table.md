# PROMPT_v3a §4.2 A2 — 离线 ckpt 重选 (composite_v3a)

- val data: `koopman_val.npz`  K=20  baseline=`checkpoints/koopman_v1_best.pth`
- 公式: composite_v3a = vel_rmse_mean × max(1, inst) × (1+2·max(0, slope-0.65)) × (1+5·max(0, deg%/100-0.18))

| ckpt | vel_rmse_mean | vel_K | u_K | v_K | inst | slope | deg% | composite_v3a |
|---|---|---|---|---|---|---|---|---|
| `koopman_v3_run01_best.pth`  ⭐ | 0.03223 | 0.04715 | 0.04630 | 0.00893 | 1.4786 | 0.8153 | 56.47 | 0.185366 |

**BEST**: `checkpoints/koopman_v3_run01_best.pth`  →  `checkpoints/koopman_v3a_reselect_best.pth`
