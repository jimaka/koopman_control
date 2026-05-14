"""Compatibility entrypoint for multistep voyage Koopman training.

The active implementation lives in ``train_multistep_intra.py`` and preserves
the current NPZ segment structure, local-frame normalization, 6D ship state,
and 4D thruster control definition.
"""

from train_multistep_intra import train


if __name__ == "__main__":
    train()
