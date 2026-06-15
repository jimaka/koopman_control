# 可执行脚本

在**仓库根目录**下运行（脚本会自动 `chdir` 到根目录并设置 `PYTHONPATH`）。

| 脚本 | 说明 |
|------|------|
| `train_v2.py` | 主训练（v2/v3/v3a），`--help` 查看参数 |
| `train_v1.py` | v1 baseline 训练 |
| `eval.py` | 量化评估与多模型对比 |
| `reselect_v3a.py` | v3a 离线 composite 重选 best ckpt |
| `data/bag_test.py` 等 | 从 rosbag 构建 / 合并 / 检查数据集 |

C++ / v4：见 `new_v4_dict_input/export_v4_onnx.py`（H=200 ONNX）、`cpp/koopman_mpc/build_v4.sh`；MPC 配置见 `cpp/koopman_control/config/mpc_config.yaml`。v3 导出见 `cpp/koopman_mpc/scripts/`。
