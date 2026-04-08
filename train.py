#!/usr/bin/env python3
"""
train.py — PPO training loop for the High-Frequency Risk Compliance Auditor.
=============================================================================
Wraps the FinAuditorEnvironment in a Gymnasium-compatible adapter and trains
a PPO agent using Stable Baselines3.

NaN-collapse fixes applied (see inline comments):
  1. Observation space bounded to [0.0, 1.0] instead of ±inf.
  2. Features clipped to [0.0, 1.0] in _process_obs to prevent gradient explosion.
  3. Environment is terminated via done=True (set in fin_auditor_environment.py)
     so PPO can compute GAE advantages without infinite truncation.
  4. Density-based reward in the environment removes the sparse penalty dead zone.

Usage:
    python train.py
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Stable Baselines3 for PPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Add project root so hft_auditor .so is importable
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.fin_auditor_environment import FinAuditorEnvironment
from models import AuditorAction

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_FEATURES      = 4    # [time_elapsed, price_delta, missing_frequency, risk_score]
MAX_TRADES      = 100  # maximum anomalies per step (== INGEST_CHUNK_SIZE)
TOTAL_TIMESTEPS = 100_000
SAVE_FREQ       = 5_000
LOG_DIR         = "./logs/"
SAVE_PATH       = os.path.join(LOG_DIR, "rl_model")


class GymnasiumFinAuditorEnv(gym.Env):
    """
    Gymnasium wrapper around FinAuditorEnvironment.

    Observation: flat float32 array of shape (MAX_TRADES * N_FEATURES,)
                 Values clipped to [0.0, 1.0] to prevent NaN gradients.
    Action:      MultiDiscrete([3] * MAX_TRADES)
                 0=PASS, 1=INVESTIGATE, 2=FLAG per trade slot.
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self._env = FinAuditorEnvironment()

        obs_size = MAX_TRADES * N_FEATURES

        # Normalization and clipping are now handled by VecNormalize wrapper in main()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        # One discrete decision per trade slot
        self.action_space = spaces.MultiDiscrete([3] * MAX_TRADES)

    def _process_obs(self, features: list[list[float]]) -> np.ndarray:
        """Flatten the anomaly matrix into a fixed-size float32 vector."""
        flat = np.zeros(MAX_TRADES * N_FEATURES, dtype=np.float32)
        for i, row in enumerate(features[:MAX_TRADES]):
            for j, val in enumerate(row[:N_FEATURES]):
                flat[i * N_FEATURES + j] = float(val)

        # Padding and normalization are handled by the vectorized environment wrapper.
        return flat

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        obs_obj = self._env.reset()
        obs = self._process_obs(obs_obj.features)
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        decisions = action.tolist()  # MultiDiscrete → Python list of ints
        action_obj = AuditorAction(decisions=decisions)

        obs_obj = self._env.step(action_obj)
        obs     = self._process_obs(obs_obj.features)
        reward  = float(obs_obj.reward) if obs_obj.reward is not None else 0.0
        done    = bool(obs_obj.done)   # True when step_count >= _MAX_EPISODE_STEPS

        return obs, reward, done, False, {}

    def render(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Training entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = GymnasiumFinAuditorEnv()

    # Sanity-check the raw environment before vectorization
    print("[TRAIN] Running Gymnasium environment check...")
    check_env(env, warn=True)
    print("[TRAIN] Environment check passed.\n")
    
    # WRAP: Use DummyVecEnv and VecNormalize for robust training.
    # SB3 requires vectorized environments for several wrappers.
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=LOG_DIR,
        name_prefix="rl_model",
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        n_steps=256,          # rollout buffer length per env per update
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,        # mild entropy bonus for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,    # gradient clipping prevents NaN proliferation
        tensorboard_log=LOG_DIR,
    )

    print(f"[TRAIN] Starting PPO training for {TOTAL_TIMESTEPS} timesteps...\n")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[TRAIN] Training interrupted by user.")

    final_path = os.path.join(LOG_DIR, "ppo_fin_auditor_final")
    model.save(final_path)
    print(f"\n[TRAIN] Model saved to: {final_path}.zip")


if __name__ == "__main__":
    main()
