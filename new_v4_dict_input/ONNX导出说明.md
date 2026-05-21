# v4 模型 ONNX 导出说明

## 1. 功能说明

`export_v4_onnx.py` 用于将 `train_v4_dict_input.py` 训练得到的 v4 checkpoint 导出为 C++ MPC 可用的 ONNX 文件，并在 **test 数据集** 上对比 PyTorch 与 ONNX 的 rollout 精度，生成 CSV/JSON 报告与对比图片（保存在独立目录）。

导出后的接口与现有 C++ 部署保持一致：

| 输入 | 形状 | 含义 |
|------|------|------|
| `state0` | `(6,)` | 初始状态 `[x, y, yaw, u, v, r]` |
| `u_seq` | `(pred_len, 4)` | 未来 pred_len 步控制量；**v4 20 s 模型为 (200, 4)** |
| `dt` | 标量 | 积分步长，默认 0.1 |

| 输出 | 形状 | 含义 |
|------|------|------|
| `states` | `(pred_len+1, 6)` | 开环 rollout 状态序列 |

> 注意：ONNX 导出时 **horizon 固定**为命令行 `--pred_len`（v4 部署默认 **200**）。须与 `cpp/koopman_control/config/mpc_config.yaml` 中 `horizon` 及 C++ `KoopmanOnnxModel::horizon()` 一致。

## 2. 环境依赖

在仓库根目录执行，需安装：

```bash
pip install torch onnx onnxruntime onnxscript
```

建议在 Docker 容器 `koopman_latest_sm120_martin` 中运行（与训练环境一致）。

## 3. 基本用法

### 3.1 使用默认 best checkpoint 导出

```bash
cd /workspace   # 仓库根目录
python3 new_v4_dict_input/export_v4_onnx.py
```

默认读取：
- checkpoint：`checkpoints/koopman_v4_best.pth`
- 输出目录：`cpp/koopman_mpc/weights/koopman_rollout.onnx`

### 3.2 指定 checkpoint 与输出目录

```bash
python3 new_v4_dict_input/export_v4_onnx.py \
  --ckpt checkpoints/koopman_v4_best.pth \
  --out_dir cpp/koopman_mpc/weights
```

### 3.3 指定某次 epoch 的 checkpoint

```bash
python3 new_v4_dict_input/export_v4_onnx.py \
  --ckpt checkpoints/run_v4_20260519_100647/koopman_v4_epoch1000.pth \
  --out_dir eval_out/v4_onnx_epoch1000
```

### 3.4 指定对比报告目录

默认写入 `eval_out/v4_onnx_compare/<tag>_<timestamp>/`，也可手动指定：

```bash
python3 new_v4_dict_input/export_v4_onnx.py \
  --ckpt checkpoints/koopman_v4_best.pth \
  --report_dir eval_out/v4_onnx_compare/manual_run
```

### 3.5 同时生成 C++ 验证用 rollout_check.npz

```bash
python3 new_v4_dict_input/export_v4_onnx.py \
  --ckpt checkpoints/koopman_v4_best.pth \
  --out_dir cpp/koopman_mpc/weights \
  --write_rollout_check
```

### 3.6 仅导出 ONNX，跳过 test 集对比

```bash
python3 new_v4_dict_input/export_v4_onnx.py --skip_test_compare
```

## 4. 输出文件

### 4.1 ONNX 部署目录（默认 `cpp/koopman_mpc/weights/`）

| 文件 | 说明 |
|------|------|
| `koopman_rollout.onnx` | C++ MPC 使用的 rollout 模型 |
| `model_meta.json` | 归一化参数、horizon、v4 模型元信息、对比结果摘要 |
| `rollout_check.npz` | 可选，供 `verify_rollout` 做 PT/ONNX 对照 |

### 4.2 精度对比报告目录（默认 `eval_out/v4_onnx_compare/<tag>_<timestamp>/`）

| 文件 | 说明 |
|------|------|
| `*_pt_onnx_summary.json` | test 集 PT vs ONNX 汇总指标 |
| `*_pt_onnx_per_step.csv` | 逐步 u/v/r 及整体误差 |
| `*_state_max_err_vs_step.png` | 全状态最大误差随 step 变化 |
| `*_u_pt_onnx_rmse_vs_step.png` | 纵向速度 u 的 PT/ONNX 误差曲线 |
| `*_v_pt_onnx_rmse_vs_step.png` | 横向速度 v 的 PT/ONNX 误差曲线 |
| `*_r_pt_onnx_rmse_vs_step.png` | 角速度 r 的 PT/ONNX 误差曲线 |
| `*_u_pt_onnx_scatter.png` | u 的 PyTorch vs ONNX 散点图 |
| `*_v_pt_onnx_scatter.png` | v 的 PyTorch vs ONNX 散点图 |
| `*_u_pt_onnx_curve_samples.png` | 6 个样本 u 曲线对比 |
| `*_v_pt_onnx_curve_samples.png` | 6 个样本 v 曲线对比 |
| `*_pt_onnx_err_hist_step20.png` | 末步 u/v/r 误差直方图 |

`model_meta.json` 中会额外记录 v4 特有字段：
- `model_class`: `HorizontalKoopmanModelV4DictInput`
- `input_mode`: `dict16_only`
- `feature_dict_atoms`: 16 阶字典名称列表
- `latent_dim` / `hidden_dim` / `clamp_pif`

## 5. 与 C++ MPC 联调

### 5.1 仅重新导出 ONNX（不重编 C++）

```bash
python3 new_v4_dict_input/export_v4_onnx.py \
  --ckpt checkpoints/koopman_v4_best.pth \
  --out_dir cpp/koopman_mpc/weights \
  --write_rollout_check
```

### 5.2 验证 ONNX rollout

若 C++ 已编译：

```bash
export LD_LIBRARY_PATH=cpp/koopman_mpc/third_party/onnxruntime/lib:$LD_LIBRARY_PATH
python3 cpp/koopman_mpc/scripts/write_rollout_check_txt.py
cpp/koopman_mpc/build/verify_rollout \
  cpp/koopman_mpc/weights/koopman_rollout.onnx \
  cpp/koopman_mpc/weights/rollout_check.npz
```

### 5.3 运行 C++ MPC smoketest

```bash
export LD_LIBRARY_PATH=cpp/koopman_mpc/third_party/onnxruntime/lib:$LD_LIBRARY_PATH
cpp/koopman_mpc/build/koopman_mpc_cpp --smoketest \
  --weights cpp/koopman_mpc/weights \
  --ref cpp/koopman_mpc/weights/cpp_test_ref.json
```

### 5.4 MPC 求解参数（导出 ONNX 之后）

ONNX 只负责动力学 rollout；MPC 优化参数在独立 YAML 中配置：

- 文件：`cpp/koopman_control/config/mpc_config.yaml`
- 加载：`loadMpcConfigFromYaml()`

| 参数 | v4 默认 | 说明 |
|------|---------|------|
| `horizon` | 200 | 须等于 `--pred_len` |
| `control_hold_steps` | 10 | 控制每 1 s 变一次（块内 u 相同） |
| `opt_control_steps` | 40 | 仅优化前 4 s（4 块） |
| `throttle_du_max` / `rudder_du_max` | 15 / 3.5 | 块间变化速率硬约束 |

详见 [`cpp/koopman_control/模型输入输出接口说明.md`](../cpp/koopman_control/模型输入输出接口说明.md) 与 [`docs/MPC使用指南.md`](../docs/MPC使用指南.md)。

> `cpp_test_ref.json` 仍由 `cpp/koopman_mpc/scripts/export_cpp_test_ref.py` 生成；
> 该脚本目前只支持 v1/v2/v3 加载。若需完整 v4 C++ 流水线，建议后续再补
> v4 版 test ref 导出脚本，或扩展原脚本支持 v4。

## 6. 成功判据

终端应出现类似输出：

```text
[OK] Saved ONNX -> cpp/koopman_mpc/weights/koopman_rollout.onnx
[OK] random-case ONNX vs PyTorch max_abs_err=4.8e-06
[OK] test-set PT vs ONNX max_abs_err=4.8e-06
[OK] compare report -> eval_out/v4_onnx_compare/v4_20260520_105500
[OK] Updated meta -> cpp/koopman_mpc/weights/model_meta.json
=== V4 ONNX EXPORT + COMPARE DONE ===
```

一般要求：
- random-case / test-set 的 `max_abs_err < 1e-4`（默认 `--atol 1e-4`）
- 无 `state_dict` 维度不匹配报错
- 对比报告目录下生成 CSV 与 PNG 图片

## 7. 常见问题

### Q1: 报错 `checkpoint model_class=... 本脚本仅支持 v4`

说明传入的不是 v4 checkpoint。请确认使用 `train_v4_dict_input.py` 训练产物，
例如 `koopman_v4_best.pth`。

### Q2: 报错 `state_dict` size mismatch

常见原因：
- checkpoint 与 `hidden_dim` / `clamp_pif` 不一致（脚本会从 ckpt 的 `args` 读取，一般不应出现）
- checkpoint 文件损坏或不完整

### Q3: 原有 `export_onnx.py` 还能用吗？

可以，但它面向 v1/v2/v3，**不能直接加载 v4**。v4 请使用本脚本。

### Q4: 重新训练后要不要重新导出？

要。每次更换 v4 checkpoint 后都应重新运行本脚本，否则 C++ MPC 仍在使用旧 ONNX。

## 8. Docker 一键示例

```bash
docker exec -it koopman_latest_sm120_martin bash -lc '
cd /workspace && \
python3 new_v4_dict_input/export_v4_onnx.py \
  --ckpt checkpoints/koopman_v4_best.pth \
  --out_dir cpp/koopman_mpc/weights \
  --write_rollout_check
'
```
