# Deep-Koopman 6-DoF 训练 / 推理

## How to train v2

```bash
# 自检（无 GPU 也可在 1 分钟内跑完）
python3 train_koopman_v2.py --smoketest

# 默认训练（速度跟踪强化版，建议 100~150 epoch）
python3 train_koopman_v2.py --config configs/koopman_v2_default.yaml

# 训练完后用现有推理脚本可视化与评估
python3 test_and_plot.py
```

训练脚本 `train_koopman_v2.py` 与现有 `koopman.py`、`test_and_plot.py`、
YAML 部署管线完全兼容（不会修改 `koopman.py` / `test_and_plot.py` / 任何
`.npz`）。详细设计与损失定义见 `PROMPT_deep_koopman_rewrite.md`。
