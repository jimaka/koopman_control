# PROMPT_v3a §4.2 A2 — 离线 ckpt 重选 (composite_v3a)

- val data: `koopman_val.npz`  K=20  baseline=`checkpoints/koopman_v1_best.pth`
- 公式: composite_v3a = vel_rmse_mean × max(1, inst) × (1+2·max(0, slope-0.65)) × (1+5·max(0, deg%/100-0.18))

| ckpt | vel_rmse_mean | vel_K | u_K | v_K | inst | slope | deg% | composite_v3a |
|---|---|---|---|---|---|---|---|---|
| `koopman_v3_run05_best.pth`  ⭐ | 0.01290 | 0.02085 | 0.01798 | 0.01055 | 1.1240 | 0.6287 | 1.09 | 0.014502 |
| `koopman_v3_run02_best.pth` | 0.01487 | 0.02295 | 0.01853 | 0.01353 | 1.1640 | 0.6584 | 0.95 | 0.017597 |
| `koopman_v3_run07_best.pth` | 0.01401 | 0.02335 | 0.01861 | 0.01411 | 1.2594 | 0.7039 | 31.29 | 0.032542 |
| `koopman_v3_run08_best.pth` | 0.02003 | 0.02996 | 0.02614 | 0.01464 | 1.1373 | 0.6424 | 33.70 | 0.040649 |
| `koopman_v3_run10_best.pth` | 0.02222 | 0.03357 | 0.03187 | 0.01055 | 1.1153 | 0.6219 | 48.54 | 0.062620 |
| `koopman_v3_run09_best.pth` | 0.02069 | 0.03426 | 0.03061 | 0.01538 | 1.1830 | 0.6530 | 66.87 | 0.084783 |
| `koopman_v3_run04_best.pth` | 0.01570 | 0.02918 | 0.02512 | 0.01485 | 2.0603 | 0.9698 | 32.83 | 0.092368 |
| `koopman_v3_run03_best.pth` | 0.04987 | 0.05511 | 0.05369 | 0.01244 | 0.9713 | 0.5497 | 70.48 | 0.180736 |
| `koopman_v3_run06_best.pth` | 0.02097 | 0.03792 | 0.03410 | 0.01659 | 1.7975 | 0.9014 | 63.34 | 0.185030 |
| `koopman_v3_run01_best.pth` | 0.03223 | 0.04715 | 0.04630 | 0.00893 | 1.4786 | 0.8153 | 56.47 | 0.185366 |

**BEST**: `checkpoints/koopman_v3_run05_best.pth`  →  `checkpoints/koopman_v3a_reselect_best.pth`
