"""Sea-trial maneuver signal generators for Koopman data collection.

Control vector matches NPZ / chassis_feedback convention (degrees for angles)::

    [port_throttle, port_angle_deg, stbd_throttle, stbd_angle_deg]

Throttle roughly in [-100, 100]; rudder typically within ±35 deg for safety.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np

Cmd = Tuple[float, float, float, float]  # port_th, port_ang, stbd_th, stbd_ang


@dataclass
class ManeuverSpec:
    name: str
    label_zh: str
    duration_s: float
    split: str  # train | val | test | supplement
    generator: str  # key into GENERATORS


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _prbs_bit(t: float, bit_duration: float, seed: int = 7) -> float:
    """Deterministic ±1 PRBS-like sequence from time index."""
    k = int(t / max(bit_duration, 1e-6))
    # xorshift-ish
    x = (k * 1103515245 + seed) & 0x7FFFFFFF
    return 1.0 if (x >> 16) & 1 else -1.0


def gen_idle(_t: float, **_kw) -> Cmd:
    return (0.0, 0.0, 0.0, 0.0)


def gen_fwd(t: float, throttle: float = 45.0, **_kw) -> Cmd:
    # Soft ramp first 5 s
    ramp = min(1.0, t / 5.0)
    th = throttle * ramp
    return (th, 0.0, th, 0.0)


def gen_astern(t: float, throttle: float = -35.0, **_kw) -> Cmd:
    ramp = min(1.0, t / 5.0)
    th = throttle * ramp
    return (th, 0.0, th, 0.0)


def gen_diff_turn(
    t: float,
    base_throttle: float = 40.0,
    delta: float = 25.0,
    rudder: float = 8.0,
    period: float = 40.0,
    **_kw,
) -> Cmd:
    """Differential thrust turn; sign flips each half-period (covers L/R)."""
    sign = 1.0 if (int(t / max(period / 2.0, 1e-6)) % 2 == 0) else -1.0
    port = base_throttle + sign * delta
    stbd = base_throttle - sign * delta
    ang = sign * rudder
    return (port, ang, stbd, -ang * 0.3)


def gen_left_turn(
    t: float,
    base_throttle: float = 40.0,
    delta: float = 28.0,
    rudder: float = 12.0,
    **_kw,
) -> Cmd:
    """Sustained left-biased differential (supplement dataset)."""
    ramp = min(1.0, t / 5.0)
    port = (base_throttle - delta) * ramp
    stbd = (base_throttle + delta) * ramp
    return (port, -rudder * ramp, stbd, rudder * 0.2 * ramp)


def gen_zigzag(
    t: float,
    throttle: float = 45.0,
    rudder_amp: float = 25.0,
    period: float = 20.0,
    **_kw,
) -> Cmd:
    """Classic zigzag: hold ±rudder for half-period each."""
    half = period / 2.0
    phase = int(t / max(half, 1e-6)) % 2
    ang = rudder_amp if phase == 0 else -rudder_amp
    return (throttle, ang, throttle, ang)


def gen_chirp(
    t: float,
    throttle: float = 40.0,
    rudder_amp: float = 20.0,
    f0: float = 0.02,
    f1: float = 0.25,
    duration: float = 1000.0,
    **_kw,
) -> Cmd:
    """Linear frequency sweep on rudder (chirp)."""
    dur = max(duration, 1.0)
    # instantaneous phase of linear chirp
    k = (f1 - f0) / dur
    phase = 2.0 * math.pi * (f0 * t + 0.5 * k * t * t)
    ang = rudder_amp * math.sin(phase)
    return (throttle, ang, throttle, ang)


def gen_prbs(
    t: float,
    throttle: float = 40.0,
    rudder_amp: float = 22.0,
    bit_duration: float = 8.0,
    throttle_dither: float = 8.0,
    **_kw,
) -> Cmd:
    bit_r = _prbs_bit(t, bit_duration, seed=11)
    bit_t = _prbs_bit(t, bit_duration * 1.37, seed=29)
    ang = rudder_amp * bit_r
    th = throttle + throttle_dither * bit_t
    return (th, ang, th, ang)


def gen_fig8(
    t: float,
    throttle: float = 42.0,
    rudder_amp: float = 28.0,
    lobe_period: float = 60.0,
    **_kw,
) -> Cmd:
    """Approximate figure-8: two opposite sustained turns."""
    lobe = int(t / max(lobe_period, 1e-6)) % 2
    ang = rudder_amp if lobe == 0 else -rudder_amp
    # slight differential to help yaw
    d = 10.0 if lobe == 0 else -10.0
    return (throttle + d, ang, throttle - d, ang)


def gen_crash_stop(
    t: float,
    cruise_throttle: float = 50.0,
    cruise_s: float = 40.0,
    reverse_throttle: float = -45.0,
    **_kw,
) -> Cmd:
    if t < cruise_s:
        ramp = min(1.0, t / 5.0)
        th = cruise_throttle * ramp
        return (th, 0.0, th, 0.0)
    # reverse / stop
    tau = t - cruise_s
    ramp = min(1.0, tau / 3.0)
    th = reverse_throttle * ramp
    return (th, 0.0, th, 0.0)


def gen_uturn(
    t: float,
    throttle: float = 35.0,
    rudder: float = 30.0,
    approach_s: float = 20.0,
    **_kw,
) -> Cmd:
    if t < approach_s:
        ramp = min(1.0, t / 5.0)
        th = throttle * ramp
        return (th, 0.0, th, 0.0)
    # sustained starboard U-turn
    return (throttle - 15.0, rudder, throttle + 15.0, rudder)


def gen_random_sailing(
    t: float,
    throttle_mean: float = 40.0,
    throttle_span: float = 15.0,
    rudder_amp: float = 18.0,
    seed: int = 42,
    **_kw,
) -> Cmd:
    """Slow random walk in throttle/rudder (piecewise constant every 5 s)."""
    slot = int(t / 5.0)
    rng = np.random.RandomState((seed + slot) & 0xFFFFFFFF)
    th = float(throttle_mean + throttle_span * (2.0 * rng.rand() - 1.0))
    ang = float(rudder_amp * (2.0 * rng.rand() - 1.0))
    d = float(8.0 * (2.0 * rng.rand() - 1.0))
    return (th + d, ang, th - d, ang)


GENERATORS: dict[str, Callable[..., Cmd]] = {
    "idle": gen_idle,
    "fwd": gen_fwd,
    "astern": gen_astern,
    "diff_turn": gen_diff_turn,
    "left_turn": gen_left_turn,
    "zigzag": gen_zigzag,
    "chirp": gen_chirp,
    "prbs": gen_prbs,
    "fig8": gen_fig8,
    "crash_stop": gen_crash_stop,
    "uturn": gen_uturn,
    "random": gen_random_sailing,
}


def apply_limits(
    cmd: Cmd,
    throttle_max: float = 100.0,
    rudder_max: float = 35.0,
) -> Cmd:
    p_th, p_a, s_th, s_a = cmd
    return (
        _clamp(p_th, -throttle_max, throttle_max),
        _clamp(p_a, -rudder_max, rudder_max),
        _clamp(s_th, -throttle_max, throttle_max),
        _clamp(s_a, -rudder_max, rudder_max),
    )


def rate_limit(
    prev: Cmd,
    target: Cmd,
    dt: float,
    d_throttle: float = 30.0,
    d_rudder: float = 20.0,
) -> Cmd:
    """Limit per-second change (smooth actuator)."""
    out = []
    limits = (d_throttle * dt, d_rudder * dt, d_throttle * dt, d_rudder * dt)
    for p, t, lim in zip(prev, target, limits):
        delta = t - p
        if abs(delta) > lim:
            delta = math.copysign(lim, delta)
        out.append(p + delta)
    return (out[0], out[1], out[2], out[3])


def sample_maneuver(
    generator: str,
    duration_s: float,
    rate_hz: float,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return times (N,) and cmds (N,4)."""
    gen = GENERATORS[generator]
    n = max(1, int(round(duration_s * rate_hz)))
    ts = np.arange(n, dtype=np.float64) / rate_hz
    cmds = np.zeros((n, 4), dtype=np.float64)
    prev: Cmd = (0.0, 0.0, 0.0, 0.0)
    dt = 1.0 / rate_hz
    kw = dict(kwargs)
    kw.setdefault("duration", duration_s)
    for i, t in enumerate(ts):
        raw = gen(float(t), **kw)
        limited = apply_limits(raw)
        prev = rate_limit(prev, limited, dt)
        cmds[i] = prev
    return ts, cmds
