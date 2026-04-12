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

        # We default to HARD, but the actual routing happens in reset()
        self.difficulty = hft_auditor.Difficulty.HARD
        self._MAX_EPISODE_STEPS = 20

        # Initialize confusion-matrix counters here so they always exist on
        # the State object — even when step() is called on a fresh env that
        # has not yet had reset() called (OpenEnv HTTP stateless mode creates
        # a new env per request, so step_handler calls step() directly).
        self._state.total_tp = 0
        self._state.total_tn = 0
        self._state.total_fp = 0
        self._state.total_fn = 0
        self._state.last_tp = 0
        self._state.last_tn = 0
        self._state.last_fp = 0
        self._state.last_fn = 0

    # FIX 1: Add *args, **kwargs to prevent TypeError when OpenEnv injects task_id
    def reset(self, *args, **kwargs) -> AuditorObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # FIX 2: Dynamically shift difficulty based on OpenEnv's requested task
        task_id = kwargs.get("task_id", os.getenv("TASK_ID", "anomaly_detection_hard")).lower()
        
        if "easy" in task_id:
            self.difficulty = hft_auditor.Difficulty.EASY
            self._MAX_EPISODE_STEPS = 5
        elif "medium" in task_id:
            self.difficulty = hft_auditor.Difficulty.MEDIUM
            self._MAX_EPISODE_STEPS = 10
        else:
            self.difficulty = hft_auditor.Difficulty.HARD
            self._MAX_EPISODE_STEPS = 20
        
        # 1. Initialize Cumulative Counters for the Grader
        self._state.total_tp = 0
        self._state.total_tn = 0
        self._state.total_fp = 0
        self._state.total_fn = 0

        self._state.last_tp = 0
        self._state.last_tn = 0
        self._state.last_fp = 0
        self._state.last_fn = 0

        # 2. Pre-generate the first batch so Step 1 actually has data!
        self.engine.generate_batch(self.difficulty, self._INGEST_CHUNK_SIZE, self.sim_time_ns)
        self.sim_time_ns += self._DELTA_MAX_NS + 1_000_000_000
        self.engine.tick(self.sim_time_ns)
        
        anomalies: list[list[float]] = self.engine.get_anomaly_matrix().tolist()

        return FinAuditorObservation(
            features=anomalies,
            message=f"Fin Auditor engine ready. {len(anomalies)} trades loaded.",
            reward=0.0,
            done=False
        )

    def step(self, action: AuditorAction) -> AuditorObservation:  # type: ignore[override]
        self._state.step_count += 1

        # 1. EVALUATE AGENT DECISIONS
        if action and action.decisions:
            action_array = np.array(action.decisions, dtype=np.uint8)
            self.engine.compute_reward(action_array)

            # ACCUMULATE metrics across the ENTIRE episode for the Grader!
            self._state.total_tp += self.engine.last_tp
            self._state.total_tn += self.engine.last_tn
            self._state.total_fp += self.engine.last_fp
            self._state.total_fn += self.engine.last_fn

            # Expose the single-batch metrics for your React dashboard
            self._state.last_tp = self.engine.last_tp
            self._state.last_tn = self.engine.last_tn
            self._state.last_fp = self.engine.last_fp
            self._state.last_fn = self.engine.last_fn

        # 2. ENGINE PROGRESSION
        self.engine.generate_batch(self.difficulty, self._INGEST_CHUNK_SIZE, self.sim_time_ns)
        self.sim_time_ns += self._DELTA_MAX_NS + 1_000_000_000
        self.engine.tick(self.sim_time_ns)

        # 3. EXTRACT NEXT MATRIX
        anomalies: list[list[float]] = self.engine.get_anomaly_matrix().tolist()
        done = self._state.step_count >= self._MAX_EPISODE_STEPS

        # 4. COMPUTE LIVE STEP REWARD from cumulative episode performance
        #    Uses same asymmetric weights as FinAuditorGrader so the dashboard
        #    value is consistent with the official final episode score.
        tp = float(self._state.total_tp)
        tn = float(self._state.total_tn)
        fp = float(self._state.total_fp)
        fn = float(self._state.total_fn)
        total = tp + tn + fp + fn

        if total > 0:
            positive = tp * 1.0 + tn * 0.1
            negative = fp * 0.1 + fn * 0.4
            raw = max(0.0, positive - negative) / (total * 1.0)
            step_reward = max(0.01, min(0.99, raw))
        else:
            step_reward = 0.01  # floor before any decisions are made

        return FinAuditorObservation(
            features=anomalies,
            message=f"Processed batch. Found {len(anomalies)} expired trades.",
            reward=step_reward,
            done=done
        )

    def close(self) -> None:
        """No-op: called by OpenEnv HTTP server after every request.

        With the factory pattern each request gets a *fresh* instance, so
        there is nothing to explicitly clean up here — the C++ engine is
        reference-counted and will be released when the Python object is GC'd.
        """
        pass

    @property
    def state(self) -> State:
        return self._state