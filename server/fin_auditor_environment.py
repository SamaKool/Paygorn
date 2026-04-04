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

    _CSV_TOTAL_ROWS: int = 10000  # internal_trades.csv has ~10k data rows

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count: int = 0
        self.engine = hft_auditor.ReconciliationEngine(self._RING_BUFFER_CAPACITY)
        self.sim_time_ns = 0
        self._ingest_offset: int = 0  # tracks rolling CSV position for continuous ingestion

    def _ingest_data_chunk(self) -> None:
        """Read the next 100-row window from the CSV (wraps around at end)."""
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "internal_trades.csv"))

        # Wrap the offset so we cycle through the full dataset indefinitely
        offset = self._ingest_offset % (self._CSV_TOTAL_ROWS - self._INGEST_CHUNK_SIZE)
        df = pd.read_csv(csv_path, skiprows=range(1, offset + 1), nrows=self._INGEST_CHUNK_SIZE)
        self._ingest_offset += self._INGEST_CHUNK_SIZE

        for row in df.itertuples(index=False):
            trade_id: int = int(str(row.trade_id), 16)
            price: int = int(float(row.amount))
            quantity: int = 1
            counterparty_id: int = int(row.counterparty_id)
            timestamp_ns: int = 0  # Pre-age all trades to T=0 so they expire on next tick

            self.engine.submit_trade(trade_id, price, quantity, counterparty_id, timestamp_ns)

    def reset(self) -> AuditorObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._ingest_offset = 0  # reset CSV cursor for fresh episode

        # Establish initial watermark at 5 seconds so T=0 trades expire immediately
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

        # Ingest the next window of trades — keeps the anomaly matrix non-empty
        # across all MAX_STEPS iterations (self, not self.auditor — fixed bug)
        self._ingest_data_chunk()

        anomalies: list[list[float]] = self.engine.get_anomaly_matrix().tolist()
        step_reward = 0.0
        max_reward = 0.001 # Prevent division by zero
        
        if anomalies and action and action.decisions:
            n = min(len(anomalies), len(action.decisions))
            missing_freq = anomalies[0][2] if anomalies else 0.0
            
            for i in range(n):
                decision = action.decisions[i]
                risk_score = anomalies[i][3]
                
                # Base reward: correctly flagging any anomaly
                if decision == 2:
                    step_reward += 1.0
                elif decision == 0:
                    step_reward -= 5.0  # harsh penalty for passing anomalies
                
                # Medium task bonus: reward precision on high-risk counterparties
                if decision == 2 and risk_score > 0.5:
                    step_reward += 0.5   
                
                # Hard task: systemic anomaly wave bonus
                if missing_freq > 0.3 and decision == 2:
                    step_reward += 1.0   
                elif missing_freq <= 0.1 and decision == 2:
                    step_reward -= 0.1   

            # Max possible per step = n * 2.5 (flag + risk bonus + systemic bonus)
            max_reward = n * 2.5
            
        # Normalize strictly to 0.0–1.0 range for the hackathon task grader
        normalized_reward = max(0.0, min(1.0, step_reward / max_reward)) if max_reward > 0 else 0.0

        return FinAuditorObservation(
            features=anomalies,
            message=f"Processed batch. Found {len(anomalies)} anomalies.",
            reward=normalized_reward,  # use the 0.0–1.0 clamped value
            done=False
        )

    @property
    def state(self) -> State:
        return self._state