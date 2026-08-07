"""v4 训练噪声增强：与 train_v2 对齐的注入逻辑冒烟。

无 torch 时仅做源码静态检查；有 torch 时验证注入行为。
"""
from __future__ import annotations

import ast
import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_V4 = ROOT / "new_v4_dict_input" / "train_v4_dict_input.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestV4NoiseAugmentSource(unittest.TestCase):
    def test_cli_and_rollout_source(self) -> None:
        src = TRAIN_V4.read_text(encoding="utf-8")
        self.assertIn('--noise_std', src)
        self.assertIn('--ctrl_noise_std', src)
        self.assertIn("noise_std: float = 0.0", src)
        self.assertIn("ctrl_noise_std: float = 0.0", src)
        self.assertIn("if model.training else 0.0", src)
        # 默认关闭（add_argument 多行写法）
        self.assertIn('default=0.0,\n        help="训练时对 t0 归一化 dyn', src)
        self.assertIn('default=0.0,\n        help="训练时对每步归一化控制', src)

    def test_ast_rollout_signature(self) -> None:
        tree = ast.parse(TRAIN_V4.read_text(encoding="utf-8"))
        fn = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "rollout_train":
                fn = node
                break
        self.assertIsNotNone(fn)
        names = [a.arg for a in fn.args.args]
        self.assertIn("noise_std", names)
        self.assertIn("ctrl_noise_std", names)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestV4NoiseAugmentRuntime(unittest.TestCase):
    def setUp(self) -> None:
        from new_v4_dict_input.train_v4_dict_input import rollout_train

        self.rollout_train = rollout_train

        class Tiny(nn.Module):
            def encode(self, x: torch.Tensor) -> torch.Tensor:
                return x

            def latent_step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
                return z + 0.1 * u[:, :3]

            def reconstruct_state(self, z: torch.Tensor) -> torch.Tensor:
                return z

        self.model = Tiny()

    def test_no_noise_deterministic(self) -> None:
        self.model.train()
        x = torch.zeros(4, 3)
        u = torch.ones(4, 5, 4)
        torch.manual_seed(0)
        a, _ = self.rollout_train(self.model, x, u, noise_std=0.0, ctrl_noise_std=0.0)
        torch.manual_seed(1)
        b, _ = self.rollout_train(self.model, x, u, noise_std=0.0, ctrl_noise_std=0.0)
        self.assertTrue(torch.allclose(a, b))

    def test_state_noise_changes_output(self) -> None:
        self.model.train()
        x = torch.zeros(8, 3)
        u = torch.zeros(8, 5, 4)
        torch.manual_seed(0)
        a, _ = self.rollout_train(self.model, x, u, noise_std=0.05, ctrl_noise_std=0.0)
        torch.manual_seed(1)
        b, _ = self.rollout_train(self.model, x, u, noise_std=0.05, ctrl_noise_std=0.0)
        self.assertFalse(torch.allclose(a, b))

    def test_ctrl_noise_changes_output(self) -> None:
        self.model.train()
        x = torch.zeros(8, 3)
        u = torch.zeros(8, 5, 4)
        torch.manual_seed(0)
        a, _ = self.rollout_train(self.model, x, u, noise_std=0.0, ctrl_noise_std=0.05)
        torch.manual_seed(1)
        b, _ = self.rollout_train(self.model, x, u, noise_std=0.0, ctrl_noise_std=0.05)
        self.assertFalse(torch.allclose(a, b))

    def test_signature(self) -> None:
        sig = inspect.signature(self.rollout_train)
        self.assertIn("noise_std", sig.parameters)
        self.assertIn("ctrl_noise_std", sig.parameters)


if __name__ == "__main__":
    unittest.main()
