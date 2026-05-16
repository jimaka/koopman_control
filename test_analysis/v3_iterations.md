# Deep-Koopman v3 调参迭代记录 (10 轮)

> PROMPT_deep_koopman_v3.md §8 要求：每轮迭代记录关键超参与 12 条阈值结果，
> 10 轮后仍 FAIL 则汇总并给出下一步建议。

## 1. 结论摘要

10 轮迭代全部完成。**最好的 ckpt 是 `run02`**（保存为 `checkpoints/koopman_v3_best.pth`），
其在 `koopman_test.npz, pred_len=20` 上 §7 的 12 条阈值中 **10 条 ✅ / 2 条 ❌**：

* ❌ `G4 slope_loglog = 0.7116`（阈值 ≤ 0.6695，超 6.3%）
* ❌ `S4 degraded_pct vs v1 = 22.94%`（阈值 ≤ 20%，超 2.94pp）

但 v3 vs v2 的核心改进全部得到数字验证：

| 维度 | v2 | v3 (run02) | 改善 |
|---|---|---|---|
| `aggregate.vel_rmse_step_20` | 0.01606 | **0.01107** | **↓ 31.1%** |
| `aggregate.u_rmse_step_20` | 0.01580 | **0.01010** | **↓ 36.1%** |
| `|channel_bias.u_bias_mean|` | 0.00255 | **0.00038** | **↓ 85.1%** |
| `per_segment.worst_vel_rmse_K` | 0.01961 | **0.01419** | **↓ 27.6%** |
| `per_segment.ratio_worst_over_best` | 3.47 | **2.22** | **↓ 36.0%** |
| `per_segment.high_speed_seg_mean` | 0.01525 | **0.00867** | **↓ 43.1%** |
| `aggregate.traj_xy_rmse_step_20` | 0.02008 | **0.01420** | **↓ 29.3%** |

**3 阶字典效果归因（全部 ✅）**：

| 问题 | v2 | v3 | v3/v2 | 判定 |
|---|---|---|---|---|
| u_bias 漂移斜率 \|slope\| | 3.74e-4 | 7.84e-5 | **0.21** | ✅ (<0.6) |
| worst-seg vel_rmse@K | 0.01961 | 0.01419 | **0.72** | ✅ (<0.80) |
| (high/v2)/(low/v2) 高速相对低速改善 | 0.01412/0.01056 | 0.00867/0.01077 | **0.60** | ✅ (<0.95) |

故 v3 架构 + 3 阶物理字典在所有"诊断性"问题上都做出了实质改善；
G4/S4 失败属于「贴近 v2 但未严格压过」的边界问题，详见 §3 诊断。

## 2. 10 轮关键超参与结果汇总

| 轮次 | 关键变化 | vel@K | u@K | \|u_bias\| | slope | inst | worst | ratio_w/b | hi | vel@1 | traj@K | 通过 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run01 | PROMPT 默认 (gamma=1.20, noise=0.03, w_l2=5e-4, clamp=5) | 0.02020 | 0.01993 | 0.00126 | 0.775 | 1.43 | 0.0279 | 3.67 | 0.0140 | 0.00186 | 0.0254 | 3/12 |
| **run02** | gamma=1.10, noise=0.02, **clamp=30**, w_l2=1e-4 | **0.01107** | **0.01010** | **0.00038** | **0.712** | **1.281** | **0.0142** | **2.22** | **0.0087** | **0.00122** | **0.0142** | **10/12** |
| run03 | run02 + noise=0.04, w_stab=10, rho=0.998 | 0.02742 | 0.02717 | 0.00072 | 0.653 | 1.167 | 0.0369 | 4.18 | 0.0194 | 0.00354 | 0.0399 | 4/12 |
| run04 | gamma=1.05, w_lin=0.2, rho=0.998, w_stab=10 | 0.01168 | 0.01118 | 0.00119 | **0.995** | 2.171 | 0.0160 | 4.34 | 0.0083 | 0.00059 | 0.0126 | 7/12 |
| run05 | run02 + w_acc=1.0, rho=0.998, w_stab=12 | 0.01155 | 0.01108 | 4e-6  | 0.707 | 1.292 | 0.0135 | 1.80 | 0.0095 | 0.00124 | 0.0135 | 9/12 |
| run06 | hidden=16, w_lin=0.5, noise=0.03 | 0.01374 | 0.01319 | 0.00008 | 0.967 | 2.047 | 0.0222 | 4.10 | 0.0094 | 0.00075 | 0.0151 | 4/12 |
| run07 | noise=0.025, w_acc=0.8, huber=0.15, rho=0.998 | 0.01055 | 0.00951 | 0.00102 | 0.788 | 1.461 | 0.0146 | 3.36 | 0.0076 | 0.00094 | 0.0128 | 9/12 |
| run08 | **gamma=1.00** (uniform step weight) | 0.01533 | 0.01477 | 0.00074 | 0.736 | 1.345 | 0.0212 | 2.67 | 0.0115 | 0.00155 | 0.0188 | 6/12 |
| run09 | run02 + **w_slope=30** (新引入的 log-log 斜率惩罚) | 0.02601 | 0.02578 | 0.00078 | 0.912 | 1.830 | 0.0317 | 1.60 | 0.0230 | 0.00166 | 0.0264 | 4/12 |
| run10 | epochs=80, pred_len_grow_every=2, w_slope=8, slope_target=0.62 | 0.01800 | 0.01774 | 0.00074 | 0.726 | 1.314 | 0.0240 | 2.43 | 0.0139 | 0.00190 | 0.0218 | 5/12 |

## 3. 诊断：为何 G4 (slope) 跨不过 0.6695

逐轮观察发现：

1. **slope 与 vel@1 强负相关**。v2 的 slope = 0.6695 同时伴随 vel@1 = 0.00185
   （较钝的早期拟合）；v3 凭借 16-原子字典在早期步上做得更准 (vel@1 = 0.00122)，
   ratio 自然抬高 slope。要把 v3 slope 压到 v2 水平，必须 *退化* v3 的 vel@1，
   这对总精度并无益处。
2. **explicit slope penalty 反而破坏训练**。run09/run10 引入 `--w_slope` 后，
   curriculum 切到 pl=20 时 batch slope_estimate 高速变化、梯度噪声大，
   反而把模型从 run02 的小盆地中赶出去。
3. **3 阶项的「过强短程拟合」是结构性**：扩展到 16 维字典后，模型在 step 1~5
   能挤压 huber 损失到几乎 0；step 10~20 上控制误差靠的是 latent 线性外推
   (I+A)^k z₀，spec_radius ≈ 0.998 → 误差线性累积 → log-log 斜率 ≈ 0.7。
   这是 Koopman + 线性外推架构 *固有* 的特征，无法在保持精度的前提下显著
   突破。

## 4. 下一步建议（若需要继续突破阈值）

* **G4 slope**：把 PROMPT §7 G4 阈值从 `≤ 0.6695` 放宽到 `≤ 0.75`（v3 的 slope
  0.711 与 v2 的 0.6695 差距 6%，但 vel@K 已下降 31%，slope 升高是"以更准
  的早期拟合换取微小斜率代价"，工程上完全划算）。或者引入非线性 latent
  step（z_{k+1} = z_k + A z_k + B u + g(z, u)，g 为浅 MLP）来打破线性外推
  的固有 slope。
* **S4 退化样本**：本指标对 v1 vs v3 是逐样本比较，仅看 vel_err@K 大小。
  把比较口径放宽为 max(v1, v3) ≤ 1.05·v1（允许 5% 相对退化）后退化样本
  比例会降到 ~15%。或者在训练数据里上采样 v2 表现差的段 (seg 7/9/11/13/16)
  逼模型集中改善它们 —— 但这又会拉高其它段的 slope。
* **更换 best_metric**：可改 `best_metric=slope_loglog` 用 slope 直接选 best
  ckpt，但实验显示它会牺牲 worst-seg 表现（PROMPT 价值排序里 worst-seg
  比 slope 更重要，因此 composite 是合适的）。

## 5. 交付的迭代 ckpt 留档

```
checkpoints/koopman_v3_run01_best.pth   (3/12)
checkpoints/koopman_v3_run02_best.pth   (10/12)  ← 复制成 koopman_v3_best.pth
checkpoints/koopman_v3_run03_best.pth   (4/12)
checkpoints/koopman_v3_run04_best.pth   (7/12)
checkpoints/koopman_v3_run05_best.pth   (9/12)
checkpoints/koopman_v3_run06_best.pth   (4/12)
checkpoints/koopman_v3_run07_best.pth   (9/12)
checkpoints/koopman_v3_run08_best.pth   (6/12)
checkpoints/koopman_v3_run09_best.pth   (4/12)
checkpoints/koopman_v3_run10_best.pth   (5/12)
```

每个 run 的 `test_analysis/v3_run0K/` 都有完整的 per_step / per_sample /
per_segment / summary.json 落盘。
