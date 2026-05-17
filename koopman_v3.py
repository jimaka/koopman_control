"""koopman_v3.py — Deep-Koopman v3: 扩展物理字典到 16 阶 (5 二次 + 11 三次)。

实现 PROMPT_deep_koopman_v3.md 第 4 节定义的 ``HorizontalKoopmanModelV3``。

关键约束（详见 PROMPT 第 3 节）：
* 不修改 ``koopman.py``。
* atom 顺序固定，YAML 反序列化必须能逐位对齐。
* encode / latent_step / reconstruct_state / spectral_radius 四个签名与 v2 等价。
* 所有字典项闭式公式实现（不引入可学权重）。
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn

from koopman import BaseKoopmanModel, res_mlp


_LOGGER = logging.getLogger(__name__)


# 16 个 atom 的固定名字顺序（与 PROMPT §4.2 完全一致；YAML / ckpt 也按此顺序）。
FEATURE_DICT_ATOMS: List[str] = [
    # quadratic (5)
    "u_abs_u", "v_abs_v", "r_abs_r", "v_times_r", "u_times_r",
    # cubic (11)
    "uvr", "u2r", "v2r", "ur2", "vr2",
    "u_vabs_v", "v_uabs_u", "r_uabs_u", "r_vabs_v",
    "uuu", "vvv",
]


def _compute_atoms_def(u: torch.Tensor, v: torch.Tensor, r: torch.Tensor) -> List[torch.Tensor]:
    """按定义计算 16 个 atom（顺序与 ``FEATURE_DICT_ATOMS`` 严格对齐）。

    输入均为同 shape 的 (...,1) 张量，返回长度 16 的列表。
    """
    abs_u = torch.abs(u)
    abs_v = torch.abs(v)
    abs_r = torch.abs(r)

    quadratic = [
        u * abs_u,          # u_abs_u  (= u·|u|)
        v * abs_v,          # v_abs_v
        r * abs_r,          # r_abs_r
        v * r,              # v_times_r
        u * r,              # u_times_r
    ]
    cubic = [
        u * v * r,          # uvr
        u * u * r,          # u2r
        v * v * r,          # v2r
        u * r * r,          # ur2
        v * r * r,          # vr2
        u * abs_v * v,      # u_vabs_v
        v * abs_u * u,      # v_uabs_u
        r * abs_u * u,      # r_uabs_u
        r * abs_v * v,      # r_vabs_v
        u * abs_u * u,      # uuu  (== u²·|u|)
        v * abs_v * v,      # vvv  (== v²·|v|)
    ]
    return quadratic + cubic


class HorizontalKoopmanModelV3(BaseKoopmanModel):
    """v3 模型：z = [state(3), 二次(5), 三次(11), hidden_mlp(hidden_dim)]。

    默认 latent_dim = 3 + 5 + 11 + 24 = 43。

    与 v2 的差别仅在 ``compute_pif_atoms`` 多了 11 项三次 atom，
    其余 (encoder MLP / A / B / latent_step / reconstruct_state) 保持不变。
    """

    def __init__(
        self,
        state_dim: int = 3,
        control_dim: int = 4,
        hidden_dim: int = 24,
        n_cubic: int = 11,
        clamp_pif: float = 5.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_dim = hidden_dim
        self.n_cubic = int(n_cubic)
        self.n_quad = 5  # 固定 5 项二次 atom
        if self.n_cubic > 11:
            raise ValueError(
                f"HorizontalKoopmanModelV3 仅实现了 11 个 cubic atom，收到 n_cubic={n_cubic}。"
                " 若要扩展请同步更新 _compute_atoms_def 与 FEATURE_DICT_ATOMS。"
            )
        if self.n_cubic < 0:
            raise ValueError("n_cubic 必须 >= 0")
        self.pif_dim = self.n_quad + self.n_cubic
        self.clamp_pif = float(clamp_pif)
        self.latent_dim = state_dim + self.pif_dim + hidden_dim

        # encoder：与 v2 同款 res_mlp，但 dropout=0（避免 encode(target)≠rollout 的污染，
        # 与 PROMPT §4 默认值一致）。
        self.encoder_mlp = res_mlp([state_dim, 64, 64, 64, hidden_dim], dropout=0.0)

        self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=True)
        self.B = nn.Linear(self.control_dim, self.latent_dim, bias=False)

        self.reset_parameters()

    # ------------------------------------------------------------------
    # 参数初始化
    # ------------------------------------------------------------------
    def reset_parameters(self) -> None:
        for m in self.encoder_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        nn.init.normal_(self.A.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.A.bias)
        nn.init.xavier_uniform_(self.B.weight, gain=0.1)

    # ------------------------------------------------------------------
    # 16 个物理字典 atom
    # ------------------------------------------------------------------
    def compute_pif_atoms(self, x: torch.Tensor) -> torch.Tensor:
        """计算 16 个物理字典 atom，返回 shape (..., pif_dim)。

        x: (..., 3) 归一化后的 [u, v, r]。
        """
        u = x[..., 0:1]
        v = x[..., 1:2]
        r = x[..., 2:3]

        atoms = _compute_atoms_def(u, v, r)
        # 按 self.n_cubic 截取 cubic 部分；前 5 项是固定 quadratic
        full = torch.cat(atoms[: self.n_quad + self.n_cubic], dim=-1)
        if self.clamp_pif > 0:
            full = torch.clamp(full, -self.clamp_pif, self.clamp_pif)
        return full

    # ------------------------------------------------------------------
    # 三大核心方法
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pif = self.compute_pif_atoms(x)        # (..., pif_dim)
        h = self.encoder_mlp(x)                # (..., hidden_dim)
        z = torch.cat([x, pif, h], dim=-1)     # (..., latent_dim)
        # 数值卫生（PROMPT §4.4）
        if z.requires_grad and not torch.jit.is_scripting():
            try:
                with torch.no_grad():
                    z_norm_max = z.norm(dim=-1).max().item()
                if z_norm_max > 50.0:
                    _LOGGER.warning(
                        "HorizontalKoopmanModelV3.encode: latent norm max=%.3g > 50, "
                        "可能 atom 溢出。", z_norm_max,
                    )
            except Exception:
                pass
        return z

    def reconstruct_state(self, z: torch.Tensor) -> torch.Tensor:
        return z[..., : self.state_dim]

    def latent_step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return z + self.A(z) + self.B(u)

    def forward(
        self,
        x_t: torch.Tensor,
        u_t: torch.Tensor,
        x_tp1: Optional[torch.Tensor] = None,
    ):
        z_t = self.encode(x_t)
        z_tp1_hat = self.latent_step(z_t, u_t)
        x_t_recon = self.reconstruct_state(z_t)
        x_tp1_hat = self.reconstruct_state(z_tp1_hat)
        if x_tp1 is not None:
            return z_t, self.encode(x_tp1), z_tp1_hat, x_t_recon, x_tp1_hat
        return z_t, z_tp1_hat, x_t_recon, x_tp1_hat

    # ------------------------------------------------------------------
    # 自检（PROMPT §4.3）
    # ------------------------------------------------------------------
    def _self_check_dict(self, tol: float = 1e-6) -> None:
        """用一组人工输入断言 16 维 atom 实现与定义吻合。

        失败立即 ``raise AssertionError``，便于 smoketest 抓取错误。
        """
        # 保存原 clamp 值临时关闭 clamp（防止 clamp 改变定义）
        original_clamp = self.clamp_pif
        try:
            self.clamp_pif = 0.0
            x = torch.tensor([[0.5, -0.3, 0.2]], dtype=torch.float64)
            atoms = self.compute_pif_atoms(x.float()).double().squeeze(0).tolist()
            # 按定义重算 expected
            u, v, r = 0.5, -0.3, 0.2
            expected = [
                u * abs(u),               # u_abs_u
                v * abs(v),               # v_abs_v
                r * abs(r),               # r_abs_r
                v * r,                    # v_times_r
                u * r,                    # u_times_r
                u * v * r,                # uvr
                u * u * r,                # u2r
                v * v * r,                # v2r
                u * r * r,                # ur2
                v * r * r,                # vr2
                u * abs(v) * v,           # u_vabs_v
                v * abs(u) * u,           # v_uabs_u
                r * abs(u) * u,           # r_uabs_u
                r * abs(v) * v,           # r_vabs_v
                u * abs(u) * u,           # uuu
                v * abs(v) * v,           # vvv
            ]
            assert len(atoms) == self.pif_dim == 16, (
                f"atom 数量不对：got={len(atoms)} pif_dim={self.pif_dim}"
            )
            for i, (got, exp, name) in enumerate(zip(atoms, expected, FEATURE_DICT_ATOMS)):
                if abs(got - exp) > tol:
                    raise AssertionError(
                        f"atom[{i}] {name}: got={got:.10g} expected={exp:.10g} diff={got - exp:.3g}"
                    )
            # 再做 batch shape 一致性 check
            xb = torch.randn(4, 3)
            atoms_b = self.compute_pif_atoms(xb)
            assert atoms_b.shape == (4, 16), f"batch shape 不对：{atoms_b.shape}"
        finally:
            self.clamp_pif = original_clamp


def smoketest_self_check() -> None:
    """模块级冒烟：实例化并跑 _self_check_dict。"""
    m = HorizontalKoopmanModelV3()
    m._self_check_dict()
    print("[koopman_v3] _self_check_dict OK; latent_dim=", m.latent_dim,
          " atoms=", FEATURE_DICT_ATOMS)


if __name__ == "__main__":
    smoketest_self_check()
