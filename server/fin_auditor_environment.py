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
from uuid import uuid4
import numpy as np
import hft_auditor
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
    _MAX_EPISODE_STEPS: int = 10

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.engine = hft_auditor.ReconciliationEngine(self._RING_BUFFER_CAPACITY)
        self.sim_time_ns = 0
        
        # 1. READ TASK_ID FROM ENVIRONMENT
        task_id = os.getenv("TASK_ID", "anomaly_detection_hard").lower()
        
        # 2. MAP TO C++ DIFFICULTY ENUM
        if "easy" in task_id:
            self.difficulty = hft_auditor.Difficulty.EASY
        elif "medium" in task_id:
            self.difficulty = hft_auditor.Difficulty.MEDIUM
        else:
            self.difficulty = hft_auditor.Difficulty.HARD

    def reset(self) -> AuditorObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # We intentionally return an empty matrix on reset to bypass the 
        # _last_reasoning scoping bug in inference.py. 
        self.sim_time_ns = self._DELTA_MAX_NS
        self.engine.tick(self._DELTA_MAX_NS)

        return FinAuditorObservation(
            features=[],
            message="Fin Auditor engine ready.",
            reward=0.0,
            done=False
        )

    def step(self, action: AuditorAction) -> AuditorObservation:  # type: ignore[override]
        self._state.step_count += 1

        # 1. REWARD CALCULATION (Using C++ Engine)
        # We must calculate reward for the PREVIOUS step's matrix before we overwrite it.
        # This uses the true False Positive / True Negative logic from Samarth's engine.
        step_reward = 0.0
        if action and action.decisions:
            # Cast actions to uint8 array for nanobind
            action_array = np.array(action.decisions, dtype=np.uint8)
            step_reward = float(self.engine.compute_reward(action_array))
            
            # Normalize the raw reward so the max possible is 1.0 per trade
            # The maximum possible reward per batch is _INGEST_CHUNK_SIZE * REWARD_TRUE_POSITIVE
            max_possible_reward = self._INGEST_CHUNK_SIZE * 1.0 
            normalized_reward = step_reward / max_possible_reward
        else:
            normalized_reward = 0.0

        # Hard clamp
        normalized_reward = max(0.0, min(1.0, normalized_reward))

        # 2. GENERATE NEW DATA (Using procedural C++ engine)
        # Replaces the old CSV ingestion logic
        self.engine.generate_batch(self.difficulty, self._INGEST_CHUNK_SIZE, self.sim_time_ns)
        
        # 3. ADVANCE TIME & EXPIRE
        # Jump time forward by 6 seconds to guarantee the batch expires immediately
        self.sim_time_ns += 6_000_000_000
        self.engine.tick(self.sim_time_ns)

        # 4. EXTRACT NEW MATRIX
        anomalies: list[list[float]] = self.engine.get_anomaly_matrix().tolist()
        total_anomalies = len(anomalies)

        done = self._state.step_count >= self._MAX_EPISODE_STEPS

        # Expose C++ tracking metrics to the Python state so inference.py can log them
        self._state.last_tp = self.engine.last_tp
        self._state.last_tn = self.engine.last_tn
        self._state.last_fp = self.engine.last_fp
        self._state.last_fn = self.engine.last_fn

        return FinAuditorObservation(
            features=anomalies,
            message=f"Processed batch. Found {total_anomalies} expired trades.",
            reward=normalized_reward,
            done=done
        )

    @property
    def state(self) -> State:
        return self._state