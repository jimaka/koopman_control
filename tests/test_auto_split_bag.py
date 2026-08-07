"""auto_split_bag 纯函数单元测试（不依赖 ROS/rosbag）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.data.auto_split_bag import (  # noqa: E402
    find_cmd_onset,
    load_phases,
    slice_phases,
)

HZ = 10.0


def _make_aligned(duration_s: float = 120.0, cmd_onset_s: float = 10.0):
    """合成 10Hz 对齐数据：cmd_onset_s 前控制为零，之后油门 40。"""
    n = int(duration_s * HZ)
    t = np.arange(n) / HZ
    cmd = np.zeros((n, 4), dtype=np.float32)
    cmd[t >= cmd_onset_s] = [40.0, 5.0, 40.0, 5.0]
    return {
        "time": t,
        "valid": np.ones(n, dtype=bool),
        "Pos": np.zeros((n, 2), dtype=np.float32),
        "Vel": np.zeros((n, 2), dtype=np.float32),
        "pqr": np.zeros((n, 1), dtype=np.float32),
        "Euler": np.zeros((n, 1), dtype=np.float32),
        "Thrusters_CMD": cmd,
    }


def _phases():
    return [
        {"name": "fwd", "generator": "fwd", "t0": 0.0, "t1": 40.0, "split": "train"},
        {"name": "zigzag", "generator": "zigzag", "t0": 40.0, "t1": 70.0, "split": "train"},
        {"name": "idle", "generator": "idle", "t0": 70.0, "t1": 75.0, "split": "train"},
        {"name": "fig8", "generator": "fig8", "t0": 75.0, "t1": 105.0, "split": "val"},
        {"name": "tiny", "generator": "fwd", "t0": 105.0, "t1": 106.0, "split": "test"},
    ]


class TestFindCmdOnset:
    def test_detects_step_onset(self):
        a = _make_aligned(cmd_onset_s=10.0)
        onset = find_cmd_onset(a["time"], a["Thrusters_CMD"], target_hz=HZ)
        assert onset == pytest.approx(10.0, abs=2.0 / HZ)

    def test_rejects_single_sample_blip(self):
        a = _make_aligned(cmd_onset_s=30.0)
        a["Thrusters_CMD"][50] = [50.0, 0.0, 50.0, 0.0]  # t=5s 单点毛刺
        onset = find_cmd_onset(a["time"], a["Thrusters_CMD"], target_hz=HZ)
        assert onset == pytest.approx(30.0, abs=2.0 / HZ)

    def test_all_zero_returns_none(self):
        a = _make_aligned()
        a["Thrusters_CMD"][:] = 0.0
        assert find_cmd_onset(a["time"], a["Thrusters_CMD"], target_hz=HZ) is None


class TestSlicePhases:
    def test_offset_slicing_and_labels(self):
        a = _make_aligned(cmd_onset_s=10.0)
        by_split, dropped = slice_phases(a, _phases(), offset_s=10.0, target_hz=HZ)

        assert [s["name"] for s in by_split["train"]] == ["fwd", "zigzag"]
        assert [s["name"] for s in by_split["val"]] == ["fig8"]
        assert by_split["test"] == []  # 1s 的 tiny 段被 min_seg_s 丢弃
        assert len(dropped) == 1 and dropped[0]["name"] == "tiny"

        fwd = by_split["train"][0]
        # 日程 [0,40) + offset 10 → bag [10,50)，10Hz 共 400 点，len=N-1 约定
        assert fwd["len"] == 399
        assert fwd["Pos"].shape == (2, 399)
        assert fwd["Euler"].shape == (3, 399)
        assert fwd["pqr"].shape == (1, 399)
        assert fwd["Thrusters_CMD"].shape == (4, 399)
        assert fwd["split"] == "train" and fwd["t0"] == 0.0
        # onset 之后切出的段指令应全为非零（对时正确性）
        assert np.all(np.abs(fwd["Thrusters_CMD"][0]) > 1.0)

    def test_idle_skipped(self):
        a = _make_aligned()
        by_split, _ = slice_phases(a, _phases(), offset_s=10.0, target_hz=HZ)
        names = [s["name"] for segs in by_split.values() for s in segs]
        assert "idle" not in names

    def test_gap_gate_drops_segment(self):
        a = _make_aligned()
        # 在 fig8 段（offset 后 bag t∈[85,115)）中部制造缺口
        a["valid"][900:910] = False
        by_split, dropped = slice_phases(a, _phases(), offset_s=10.0, target_hz=HZ)
        assert by_split["val"] == []
        assert any(d["name"] == "fig8" and "缺口" in d["reason"] for d in dropped)
        # train 段不受影响
        assert len(by_split["train"]) == 2

    def test_out_of_range_phase_dropped(self):
        a = _make_aligned(duration_s=60.0)  # bag 只有 60s
        by_split, dropped = slice_phases(a, _phases(), offset_s=0.0, target_hz=HZ)
        assert any(d["reason"] == "超出 bag 时间范围" for d in dropped)


class TestLoadPhases:
    def test_profile_rebuild(self):
        phases = load_phases(profile="smoke")
        assert phases[0]["name"] == "fwd"
        assert any(p["generator"] == "idle" for p in phases)
        assert {p["split"] for p in phases} >= {"train", "val", "test"}

    def test_schedule_json(self, tmp_path):
        import json

        payload = {"profile": "smoke", "phases": [
            {"name": "fwd", "generator": "fwd", "t0": 0.0, "t1": 40.0, "split": "train"},
        ]}
        p = tmp_path / "schedule.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        phases = load_phases(schedule_json=str(p))
        assert len(phases) == 1 and phases[0]["name"] == "fwd"

    def test_requires_one_source(self):
        with pytest.raises(ValueError):
            load_phases()
