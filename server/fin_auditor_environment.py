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
import pandas as pd
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
    _INGEST_CHUNK_SIZE: int = 100
    _DELTA_MAX_NS: int = 5_000_000_000

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count: int = 0
        self.engine = hft_auditor.ReconciliationEngine(self._RING_BUFFER_CAPACITY)
        self.sim_time_ns = 0

    def _ingest_data_chunk(self) -> None:
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "internal_trades.csv"))
        # Fix 1: Read only 100 rows
        df = pd.read_csv(csv_path, nrows=self._INGEST_CHUNK_SIZE)

        # Fix 1: Use itertuples for massive speedup
        for row in df.itertuples(index=False):
            trade_id: int = int(str(row.trade_id), 16)
            price: int = int(float(row.amount))
            quantity: int = 1
            counterparty_id: int = int(row.counterparty_id)
            timestamp_ns: int = 0  # Fix 2: Pre-age all trades to T=0

            self.engine.submit_trade(trade_id, price, quantity, counterparty_id, timestamp_ns)

    def reset(self) -> AuditorObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        
        # Fix 2: Establish initial watermark at 5 seconds so T=0 trades expire immediately
        self.sim_time_ns = self._DELTA_MAX_NS
        self.engine.tick(self._DELTA_MAX_NS)

        self._ingest_data_chunk()

        initial_features = self.engine.get_anomaly_matrix().tolist()

        return FinAuditorObservation(
            features=initial_features,
            message="Fin Auditor engine ready.",
            reward=0.0,
            done=False
        )

    def step(self, action: AuditorAction) -> AuditorObservation:  # type: ignore[override]
        self._state.step_count += 1
        self.sim_time_ns += 100_000_000
        self.engine.tick(self.sim_time_ns)

        anomalies: list[list[float]] = self.engine.get_anomaly_matrix().tolist()
        step_reward = 0.0
        
        if anomalies and action and action.decisions:
            for i in range(min(len(anomalies), len(action.decisions))):
                decision = action.decisions[i]
                if decision == 2:    
                    step_reward += 1.0
                elif decision == 0:  
                    step_reward -= 5.0

        return FinAuditorObservation(
            features=anomalies,
            message=f"Processed batch. Found {len(anomalies)} anomalies.",
            reward=step_reward,
            done=False
        )

    @property
    def state(self) -> State:
        return self._state