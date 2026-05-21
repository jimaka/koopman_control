# v4 dict-input scripts

该目录是新增脚本目录，不改动原有 `koopman/` 与 `scripts/` 下脚本。

## 文件

- `model_v4_dict_input.py`：16 阶字典作为主输入的 Koopman 模型。
- `train_v4_dict_input.py`：参考 `train_v2.py` 重写的训练脚本（独立版本）。
- `eval_v4_dict_input.py`：v4 单模型验证脚本（默认使用 `data/koopman_test.npz`）。
- `export_v4_onnx.py`：v4 checkpoint 导出 ONNX（供 C++ MPC 使用）。
- `ONNX导出说明.md`：v4 ONNX 导出中文使用文档。
- `run_v4_in_docker.sh`：一键 Docker 启动脚本（自动探测路径并按需同步）。
- `compare_mpc_tracking.py`：多模型 MPC 跟踪精度比对脚本。

## Docker 运行约束

按你的要求，仅在容器 `koopman_latest_sm120_martin` 中运行：

推荐使用一键脚本：

```bash
bash new_v4_dict_input/run_v4_in_docker.sh --help-train
bash new_v4_dict_input/run_v4_in_docker.sh --smoketest
bash new_v4_dict_input/run_v4_in_docker.sh --train -- --epochs 60 --run_tag v4_dict_input
```

该脚本会：
- 检查容器 `koopman_latest_sm120_martin` 是否存在且运行中；
- 自动探测容器内工程目录（优先已存在 `new_v4_dict_input` 的路径）；
- 若容器内缺少 `new_v4_dict_input`，自动从宿主机同步后再执行。

也可手动执行：

```bash
docker exec -it koopman_latest_sm120_martin bash -lc '
cd /workspace && \
python3 new_v4_dict_input/train_v4_dict_input.py --help
'
```

训练示例（**20 s 预测**，dt=0.1 → 200 步；**8GB 显存** 推荐配置）：

```bash
docker exec -it koopman_latest_sm120_martin bash -lc '
cd /workspace && \
python3 new_v4_dict_input/train_v4_dict_input.py \
  --run_tag v4_dict_input_20s \
  --pred_time_sec 20 \
  --pred_time_start_sec 2 \
  --epochs 120 \
  --batch_size 512 \
  --grad_accum_steps 2 \
  --val_batch_size 128 \
  --val_max_samples 512 \
  --hidden_dim 32 \
  --w_recon 0.5
'
```

| 显存 | 建议 `batch_size` | `grad_accum_steps` | 等效 batch |
|------|-------------------|--------------------|------------|
| 8 GB | 512 | 2 | 1024 |
| 8 GB（仍 OOM，pred_len 接近 200 时） | 384 | 2 | 768 |
| 8 GB（curriculum 前期 pl≤40） | 512 | 2 | 1024 |

也可直接指定步数：`--pred_len_max 200 --pred_len_start 20`。

快速冒烟：

```bash
docker exec -it koopman_latest_sm120_martin bash -lc '
cd /workspace && \
python3 new_v4_dict_input/train_v4_dict_input.py --smoketest
'
```

test 集验证（v4）：

```bash
docker exec -it koopman_latest_sm120_martin bash -lc '
cd /workspace && \
python3 new_v4_dict_input/eval_v4_dict_input.py \
  --ckpt checkpoints/koopman_v4_best.pth \
  --data data/koopman_test.npz \
  --pred_len 20 \
  --out_dir eval_out/v4_test
'
```

验证脚本会额外生成作图结果（尤其用于横/纵向速度比对）：
- `*_u_rmse_vs_step.png` / `*_v_rmse_vs_step.png`：`u`、`v` 分开绘制的逐步 RMSE 曲线；
- `*_u_scatter_compare.png` / `*_v_scatter_compare.png`：`u`、`v` 分开绘制的 GT vs Pred 散点对比（含 `y=x` 参考线）；
- `*_u_curve_compare.png` / `*_v_curve_compare.png`：按误差分位选取 6 个样本，分别对 `u`、`v` 进行预测曲线对比。

ONNX 导出（v4 → C++ MPC）：

```bash
docker exec -it koopman_latest_sm120_martin bash -lc '
cd /workspace && \
python3 new_v4_dict_input/export_v4_onnx.py \
  --ckpt checkpoints/koopman_v4_best.pth \
  --pred_len 200 \
  --out_dir cpp/koopman_mpc/weights \
  --data data/koopman_test.npz \
  --write_rollout_check
'
```

导出后 MPC 参数见 `cpp/koopman_control/config/mpc_config.yaml`（`control_hold_steps`、`throttle_du_max` 等）；接口说明见 [`../cpp/koopman_control/模型输入输出接口说明.md`](../cpp/koopman_control/模型输入输出接口说明.md)。

导出后会在 `eval_out/v4_onnx_compare/<tag>_<timestamp>/` 生成 PT vs ONNX 精度对比 CSV/JSON 与 u/v 等对比图。

详细说明见 [ONNX导出说明.md](./ONNX导出说明.md)。

多模型跟踪精度比对（MPC）：

```bash
docker exec -it koopman_latest_sm120_martin bash -lc '
cd /workspace && \
python3 new_v4_dict_input/compare_mpc_tracking.py \
  --models checkpoints/koopman_v2_best.pth:v2 checkpoints/koopman_v3a_best.pth:v3a checkpoints/koopman_v4_best.pth:v4 \
  --ref segment --data data/koopman_test.npz --segment 0 --steps 120 \
  --out_dir eval_out/mpc_compare_seg0
'
```
