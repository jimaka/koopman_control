# Koopman MPC — C++ 实现（ONNX Runtime）

基于 **ONNX Runtime** 的船舶航迹跟踪 MPC，动力学 rollout 与 Python 版 `koopman/export/rollout.py` 对齐。

## 目录结构

```
cpp/koopman_mpc/
├── CMakeLists.txt
├── build.sh                 # 下载 ORT、导出 ONNX、编译、全流程验证
├── include/                 # ONNX 模型 + MPC
├── src/
├── tools/verify_rollout.cpp # rollout 数值对照
├── scripts/
│   ├── export_onnx.py       # .pth → .onnx + PT/ONNX 精度验证
│   ├── export_cpp_test_ref.py
│   └── verify_pipeline.py   # 端到端复验
├── third_party/onnxruntime/ # build.sh 自动下载（gitignore）
└── weights/                 # 生成物（gitignore）
    ├── koopman_rollout.onnx
    └── ...
```

## 依赖

- C++17（g++）、CMake ≥ 3.18
- Python 3 + `torch`、`onnx`、`onnxruntime`、`onnxscript`（导出与验证）

## 构建与验证

在仓库根目录：

```bash
bash cpp/koopman_mpc/build.sh
```

脚本将依次：

1. 下载 ONNX Runtime C++ 1.26
2. 从 checkpoint 导出 `weights/koopman_rollout.onnx`（PT vs ONNX 误差 &lt; 1e-4）
3. 编译 `koopman_mpc_cpp`、`verify_rollout`
4. C++ rollout 与 Python 对照（&lt; 1e-3）
5. MPC 冒烟测试

仅复验 Python + 已编译 C++：

```bash
python3 cpp/koopman_mpc/scripts/verify_pipeline.py
```

## 运行

```bash
export LD_LIBRARY_PATH=cpp/koopman_mpc/third_party/onnxruntime/lib:$LD_LIBRARY_PATH
./cpp/koopman_mpc/build/koopman_mpc_cpp \
    --weights cpp/koopman_mpc/weights \
    --ref cpp/koopman_mpc/weights/cpp_test_ref.json \
    --steps 40 --horizon 20 --opt_iters 25
```

| 参数 | 说明 |
|------|------|
| `--weights` | 含 `koopman_rollout.onnx` 的目录 |
| `--ref` | `export_cpp_test_ref.py` 生成的文本航迹 |
| `--horizon` | 须为 20（与 ONNX 导出一致） |
| `--smoketest` | 快速自检 |

## 与 Python 版关系

| 组件 | Python | C++ |
|------|--------|-----|
| 动力学 | `KoopmanRollout` | `KoopmanOnnxModel`（ONNX） |
| 优化 | `torch.optim.Adam` + autograd | Adam + 数值梯度（ONNX 前向） |
| 导出 | `scripts/export_onnx.py` | `build.sh` 调用 |

重新训练后请重新运行 `build.sh` 更新 `koopman_rollout.onnx`。
