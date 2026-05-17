# Deep-Koopman 速度跟踪

Koopman 物理先验模型与训练 / 评估管线。

**完整项目介绍与使用流程**见 [`项目指南.md`](./项目指南.md)。

## 目录结构

| 类型 | 文件 |
|------|------|
| 模型 | `koopman.py`（v1/v2）、`koopman_v3.py`（v3 16 阶字典） |
| 训练 | `train_multistep_voyage.py`（v1）、`train_koopman_v2.py`（v2/v3/v3a） |
| 验证 | `eval_koopman.py` |
| MPC 航迹跟踪 | `mpc_koopman.py`, `run_mpc_tracking.py` |
| v3a 辅助 | `scripts/reselect_best_v3a.py`（离线 composite_v3a 重选 best ckpt） |
| 数据处理 | `bag_test.py`、`extract_left_turn.py`、`merge_npz.py`、`split_high_density_bag.py`、`check_dataset.py` |
| 数据集 | `koopman_train_merged.npz`、`koopman_val.npz`、`koopman_test.npz` 等 `koopman_*.npz`、`sim_10HZ.npz`、`test_ds/koopman_test_dataset.npz` |

## 训练

```bash
# v2
python3 train_koopman_v2.py --model v2 --run_tag v2 --epochs 60

# v3
python3 train_koopman_v2.py --model v3 --run_tag v3 --epochs 60

# v3a（Plan-A：w_bias_v + composite_v3a + seg_resample）
python3 train_koopman_v2.py --model v3 --run_tag v3a \
    --w_bias_v 100.0 --best_metric composite_v3a \
    --seg_resample u_var_r_var --baseline_ckpt checkpoints/koopman_v1_best.pth

# v1 baseline
python3 train_multistep_voyage.py

# 冒烟自测
python3 train_koopman_v2.py --smoketest
```

训练结束后 best 权重写入 `checkpoints/koopman_{v2,v3,v3a}_best.pth`。

## 验证

```bash
# 单模型
python3 eval_koopman.py --ckpt checkpoints/koopman_v3a_best.pth \
    --data koopman_test.npz --pred_len 20 --tag v3a \
    --baseline_ckpt checkpoints/koopman_v1_best.pth --out_dir eval_out/v3a

# 多模型对比
python3 eval_koopman.py --compare \
    checkpoints/koopman_v1_best.pth:v1 \
    checkpoints/koopman_v2_best.pth:v2 \
    checkpoints/koopman_v3_best.pth:v3 \
    checkpoints/koopman_v3a_best.pth:v3a \
    --data koopman_test.npz --pred_len 20 --out_dir eval_out/compare

python3 eval_koopman.py --smoketest
```

## MPC 航迹跟踪

```bash
python3 run_mpc_tracking.py --ckpt checkpoints/koopman_v3a_best.pth \
    --data koopman_test.npz --segment 0 --steps 150 --out_dir eval_out/mpc
```

详见 [`项目指南.md`](./项目指南.md) 阶段 D。

**C++ 版**（`cpp/koopman_mpc/`，LibTorch）：`bash cpp/koopman_mpc/build.sh` 后运行 `build/koopman_mpc_cpp`。

## 数据处理

```bash
# 从 rosbag 生成 npz（需 ROS 环境）
python3 bag_test.py
python3 extract_left_turn.py
python3 split_high_density_bag.py

# 合并 / 检查
python3 merge_npz.py
python3 check_dataset.py --npz koopman_train_merged.npz
```
