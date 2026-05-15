# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Physics-Informed Koopman Operator model for ship/vessel dynamics prediction. A PyTorch-based ML pipeline for learning a linear Koopman operator representation of nonlinear ship motion (surge, sway, yaw rate). All code is flat Python scripts in the repository root; there is no package structure.

### Key scripts

| Script | Purpose | Example |
|---|---|---|
| `train_multistep_voyage.py` | Train the Koopman model | `python3 train_multistep_voyage.py --epochs 150` |
| `test_and_plot.py` | Evaluate model & generate plots | `python3 test_and_plot.py` |
| `check_dataset.py` | Visualize dataset segments | `python3 check_dataset.py --data koopman_train_merged.npz --seg 0` |
| `koopman.py` | Model architecture (imported, not run directly) | — |

### Running in this environment

- **No GPU available** in Cloud Agent VMs. All scripts fall back to CPU automatically (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`). Training is slow on CPU; use small `--epochs` and `--batch_size` for quick validation (e.g. `--epochs 2 --batch_size 128 --num_workers 2 --prefetch 2`).
- Pre-processed `.npz` datasets and pre-trained `.pth` checkpoints are committed to the repo, so the full pipeline (train + evaluate) works out of the box without ROS/rosbag.
- Training writes logs to `logs/` and TensorBoard events inside `logs/tensorboard_*`. Checkpoints save to `checkpoints/`.
- `test_and_plot.py` reads `checkpoints/koopman_best.pth` and `koopman_test.npz`; generates plots in `test_analysis/`.
- `auto_tuner.py` requires Docker and a Gemini API key — not expected to work in Cloud Agent VMs.
- There are no automated test suites (no pytest/unittest). Validation is done by running training + evaluation and checking the output metrics and generated plots.
- There is no linter configuration. Standard `python3 -m py_compile <file>` can be used to check syntax.
