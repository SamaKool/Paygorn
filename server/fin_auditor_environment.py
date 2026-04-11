# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Fin Auditor Environment Implementation.
Wraps the compiled C++ ``hft_auditor.ReconciliationEngine``.
"""

import os
import sys
import glob
import importlib.util
from uuid import uuid4
import numpy as np

# ── Native Engine Bridge ─────────────────────────────────────────────────────
def _load_native_engine():
    """Surgically discovers and loads the compiled C++ binary (.so or .pyd)."""
    try:
        import hft_auditor
        return hft_auditor
    except ImportError:
        pass

    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    _ROOT_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
    
    patterns = [
        os.path.join(_ROOT_DIR, "hft_auditor*.pyd"),
        os.path.join(_ROOT_DIR, "hft_auditor*.so"),
        os.path.join(_ROOT_DIR, "hf auditor/build/**/hft_auditor*.pyd"),
        os.path.join(_ROOT_DIR, "hf auditor/build/**/hft_auditor*.so"),
    ]
    
    lib_files = []
    for p in patterns:
        lib_files.extend(glob.glob(p, recursive=True))
    
    if not lib_files:
        return None
        
    try:
        lib_path = lib_files[0]
        spec = importlib.util.spec_from_file_location("hft_auditor", lib_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules["hft_auditor"] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"[CRITICAL] Native loading failed: {e}")
        return None

hft_auditor = _load_native_engine()
# ─────────────────────────────────────────────────────────────────────────────

from typing import Any, Dict, Optional
from pydantic import Field

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import AuditorAction, AuditorObservation

class FinAuditorObservation(AuditorObservation):
    model_config = AuditorObservation.model_config
    done: bool = Field(default=False)
    reward: Optional[float] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FinAuditorEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    _RING_BUFFER_CAPACITY: int = 1_048_576
    _INGEST_CHUNK_SIZE: int = 40
    _DELTA_MAX_NS: int = 5_000_000_000

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.engine = hft_auditor.ReconciliationEngine(self._RING_BUFFER_CAPACITY)
        self.sim_time_ns = 0
        
        task_id = os.getenv("TASK_ID", "anomaly_detection_hard").lower()
        
        if "easy" in task_id:
            self.difficulty = hft_auditor.Difficulty.EASY
            self._MAX_EPISODE_STEPS = 5
        elif "medium" in task_id:
            self.difficulty = hft_auditor.Difficulty.MEDIUM
            self._MAX_EPISODE_STEPS = 10
        else:
            self.difficulty = hft_auditor.Difficulty.HARD
            self._MAX_EPISODE_STEPS = 20

    def reset(self) -> AuditorObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.sim_time_ns += self._DELTA_MAX_NS
        self.engine.tick(self.sim_time_ns)

        return FinAuditorObservation(
            features=[],
            message="Fin Auditor engine ready.",
            reward=0.01,  # Safe minimum floor, not divided
            done=False
        )

    def step(self, action: AuditorAction) -> AuditorObservation:  # type: ignore[override]
        self._state.step_count += 1
        
        # 1. BASE REWARD CALCULATION
        if action and action.decisions:
            action_array = np.array(action.decisions, dtype=np.uint8)
            raw_reward = float(self.engine.compute_reward(action_array))
            
            # Map raw engine bounds [-4.0, 40.0] -> [0.0, 1.0]
            normalized_raw = (raw_reward + 4.0) / 44.0
            
            # 2. THE FIX: CLAMP AND DISTRIBUTE
            # We clamp to 0.99 so even a "perfect" sum stays under 1.0
            # Dividing by MAX_EPISODE_STEPS ensures the total sum is strictly in (0, 1)
            clamped_val = max(0.01, min(0.99, normalized_raw))
            step_reward = clamped_val / self._MAX_EPISODE_STEPS
        else:
            # Default for empty decisions: a tiny fraction of the floor
            step_reward = 0.01 / self._MAX_EPISODE_STEPS

        # 3. ENGINE PROGRESSION (Keep original logic)
        self.engine.generate_batch(self.difficulty, self._INGEST_CHUNK_SIZE, self.sim_time_ns)
        self.sim_time_ns += 6_000_000_000
        self.engine.tick(self.sim_time_ns)

        anomalies: list[list[float]] = self.engine.get_anomaly_matrix().tolist()
        total_anomalies = len(anomalies)
        done = self._state.step_count >= self._MAX_EPISODE_STEPS

        # Update metrics for the dashboard
        self._state.last_tp = self.engine.last_tp
        self._state.last_tn = self.engine.last_tn
        self._state.last_fp = self.engine.last_fp
        self._state.last_fn = self.engine.last_fn

        return FinAuditorObservation(
            features=anomalies,
            message=f"Processed batch. Found {total_anomalies} expired trades.",
            reward=step_reward,  
            done=done
        )

    @property
    def state(self) -> State:
        return self._state