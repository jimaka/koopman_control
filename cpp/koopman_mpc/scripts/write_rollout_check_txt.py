#!/usr/bin/env python3
"""将 rollout_check.npz 转为 verify_rollout 可读的 .txt。"""
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
npz = os.path.join(REPO, "cpp/koopman_mpc/weights/rollout_check.npz")
txt = npz + ".txt"
d = np.load(npz)
s0 = d["state0"]
u = d["u_seq"]
states = d["states"]
H = u.shape[0]
with open(txt, "w") as f:
    f.write(" ".join(f"{x:.9g}" for x in s0) + "\n")
    f.write(f"{H}\n")
    f.write(" ".join(f"{x:.9g}" for x in u.reshape(-1)) + "\n")
    f.write(f"{states.shape[0]} {states.shape[1]}\n")
    f.write(" ".join(f"{x:.9g}" for x in states.reshape(-1)) + "\n")
print("Wrote", txt)
