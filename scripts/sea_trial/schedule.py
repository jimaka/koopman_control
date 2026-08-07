"""Sea-trial schedules aligned with scripts/data bag post-processing timelines.

Profiles
--------
- ``dense``   : 25200 s (~7 h) — matches ``split_high_density_bag.py``
- ``standard``: 15000 s (~4.1 h) — matches ``bag_test.py``
- ``smoke``   : short dry-run (~8 min) covering every maneuver type once
- ``left_turn``: supplementary left-turn block (200 s chunks)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional

from scripts.sea_trial.maneuvers import ManeuverSpec

ProfileName = Literal["dense", "standard", "smoke", "left_turn"]


@dataclass
class Phase:
    name: str
    label_zh: str
    generator: str
    t0: float
    t1: float
    split: str

    @property
    def duration_s(self) -> float:
        return float(self.t1 - self.t0)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["duration_s"] = self.duration_s
        return d


def _append_repeats(
    phases: List[Phase],
    name: str,
    label_zh: str,
    generator: str,
    t_start: float,
    t_end: float,
    seg_len: float,
    split: str,
) -> float:
    t = float(t_start)
    while t + 1e-9 < t_end:
        t1 = min(t + seg_len, t_end)
        phases.append(Phase(name, label_zh, generator, t, t1, split))
        t = t1
    return float(t_end)


def build_dense_schedule() -> List[Phase]:
    """Match split_high_density_bag.generate_exact_phases()."""
    phases: List[Phase] = []
    _append_repeats(phases, "fwd", "直航", "fwd", 0, 900, 90, "train")
    _append_repeats(phases, "astern", "倒车", "astern", 900, 1500, 100, "train")
    _append_repeats(phases, "diff_turn", "差速转向", "diff_turn", 1500, 3300, 200, "train")
    _append_repeats(phases, "zigzag", "Z字操舵", "zigzag", 3300, 15900, 200, "train")
    phases.append(Phase("chirp", "扫频", "chirp", 15900, 16900, "train"))
    phases.append(Phase("prbs", "PRBS", "prbs", 16900, 18000, "train"))
    phases.append(Phase("fig8", "8字航行", "fig8", 18000, 19200, "val"))
    phases.append(Phase("crash_stop", "急停", "crash_stop", 19200, 20400, "val"))
    phases.append(Phase("uturn", "U型转弯", "uturn", 20400, 21600, "val"))
    _append_repeats(phases, "random", "自由随机航行", "random", 21600, 25200, 200, "test")
    return phases


def build_standard_schedule() -> List[Phase]:
    """Match bag_test.generate_exact_phases()."""
    phases: List[Phase] = []
    _append_repeats(phases, "fwd", "直航", "fwd", 0, 600, 60, "train")
    _append_repeats(phases, "astern", "倒车", "astern", 600, 1000, 100, "train")
    _append_repeats(phases, "diff_turn", "差速转向", "diff_turn", 1000, 2000, 200, "train")
    _append_repeats(phases, "zigzag", "Z字操舵", "zigzag", 2000, 9000, 200, "train")
    phases.append(Phase("chirp", "扫频", "chirp", 9000, 10000, "train"))
    phases.append(Phase("prbs", "PRBS", "prbs", 10000, 11000, "train"))
    phases.append(Phase("fig8", "8字航行", "fig8", 11000, 11700, "val"))
    phases.append(Phase("crash_stop", "急停", "crash_stop", 11700, 12400, "val"))
    phases.append(Phase("uturn", "U型转弯", "uturn", 12400, 13000, "val"))
    _append_repeats(phases, "random", "自由随机航行", "random", 13000, 15000, 200, "test")
    return phases


def build_smoke_schedule() -> List[Phase]:
    """One short block per maneuver type (~8 min)."""
    specs = [
        ("fwd", "直航", "fwd", 40, "train"),
        ("astern", "倒车", "astern", 30, "train"),
        ("diff_turn", "差速转向", "diff_turn", 40, "train"),
        ("left_turn", "左转差速", "left_turn", 40, "supplement"),
        ("zigzag", "Z字操舵", "zigzag", 40, "train"),
        ("chirp", "扫频", "chirp", 60, "train"),
        ("prbs", "PRBS", "prbs", 60, "train"),
        ("fig8", "8字航行", "fig8", 60, "val"),
        ("crash_stop", "急停", "crash_stop", 50, "val"),
        ("uturn", "U型转弯", "uturn", 50, "val"),
        ("random", "自由随机航行", "random", 40, "test"),
    ]
    phases: List[Phase] = []
    t = 0.0
    for name, zh, gen, dur, split in specs:
        phases.append(Phase(name, zh, gen, t, t + dur, split))
        t += dur
        # short idle gap for safety / annotation
        phases.append(Phase("idle", "停泊间隙", "idle", t, t + 5.0, split))
        t += 5.0
    return phases


def build_left_turn_schedule(n_segments: int = 10, seg_len: float = 200.0) -> List[Phase]:
    phases: List[Phase] = []
    t = 0.0
    for _ in range(n_segments):
        phases.append(Phase("left_turn", "左转差速", "left_turn", t, t + seg_len, "supplement"))
        t += seg_len
    return phases


def build_schedule(profile: ProfileName, **kwargs) -> List[Phase]:
    if profile == "dense":
        return build_dense_schedule()
    if profile == "standard":
        return build_standard_schedule()
    if profile == "smoke":
        return build_smoke_schedule()
    if profile == "left_turn":
        return build_left_turn_schedule(
            n_segments=int(kwargs.get("n_segments", 10)),
            seg_len=float(kwargs.get("seg_len", 200.0)),
        )
    raise ValueError(f"unknown profile: {profile}")


def schedule_summary(phases: List[Phase]) -> Dict:
    total = phases[-1].t1 if phases else 0.0
    by_name: Dict[str, float] = {}
    by_split: Dict[str, float] = {}
    for p in phases:
        by_name[p.name] = by_name.get(p.name, 0.0) + p.duration_s
        by_split[p.split] = by_split.get(p.split, 0.0) + p.duration_s
    return {
        "n_phases": len(phases),
        "total_s": total,
        "total_h": total / 3600.0,
        "by_name_s": by_name,
        "by_split_s": by_split,
    }


# Human-readable catalog (for docs / --list)
MANEUVER_CATALOG: List[ManeuverSpec] = [
    ManeuverSpec("fwd", "直航", 90, "train", "fwd"),
    ManeuverSpec("astern", "倒车", 100, "train", "astern"),
    ManeuverSpec("diff_turn", "差速转向", 200, "train", "diff_turn"),
    ManeuverSpec("zigzag", "Z字操舵", 200, "train", "zigzag"),
    ManeuverSpec("chirp", "扫频 Chirp", 1000, "train", "chirp"),
    ManeuverSpec("prbs", "PRBS 伪随机", 1100, "train", "prbs"),
    ManeuverSpec("fig8", "8字航行", 1200, "val", "fig8"),
    ManeuverSpec("crash_stop", "急停", 1200, "val", "crash_stop"),
    ManeuverSpec("uturn", "U型转弯", 1200, "val", "uturn"),
    ManeuverSpec("random", "自由随机航行", 200, "test", "random"),
    ManeuverSpec("left_turn", "左转差速补充", 200, "supplement", "left_turn"),
]
