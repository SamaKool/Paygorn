# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Fin Auditor Environment Implementation.

Wraps the compiled C++ ``hft_auditor.ReconciliationEngine`` and exposes it
as an OpenEnv ``Environment`` so Uvicorn / the existing HTTP/WebSocket server
infrastructure can drive it without modification.
"""

import os
import time
from pathlib import Path
from uuid import uuid4

import pandas as pd

import hft_auditor

from typing import Any, Dict, Optional

from pydantic import Field

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import AuditorAction, AuditorObservation


class FinAuditorObservation(AuditorObservation):
    """AuditorObservation extended with the openenv serialization contract.

    ``openenv.serialize_observation`` unconditionally reads ``obs.reward``,
    ``obs.done``, and ``obs.metadata``.  Since we cannot modify ``models.py``
    (Sacred Contract), we extend here inside the environment module.
    """

    model_config = AuditorObservation.model_config  # inherit forbid-extra etc.

    done: bool = Field(default=False)
    reward: Optional[float] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FinAuditorEnvironment(Environment):
    """
    Production environment backed by the compiled C++ reconciliation engine.

    Each instance owns its own ``ReconciliationEngine`` so that concurrent
    WebSocket sessions remain fully isolated (see ``SUPPORTS_CONCURRENT_SESSIONS``
    below).

    Example::

        env = FinAuditorEnvironment()
        obs = env.reset()          # returns AuditorObservation with initial features
        obs = env.step(AuditorAction(decisions=[0, 1, 2, ...]))
    """

    # Enable concurrent WebSocket sessions.
    # Each connecting client gets its own FinAuditorEnvironment instance,
    # and therefore its own independent ReconciliationEngine state.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Ring-buffer capacity passed to the C++ engine (power-of-two, 2^20 slots).
    _RING_BUFFER_CAPACITY: int = 1_048_576

    def __init__(self) -> None:
        """Initialise state and spin up the real C++ reconciliation engine."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count: int = 0

        # Instantiate the compiled C++ engine.
        self.engine = hft_auditor.ReconciliationEngine(self._RING_BUFFER_CAPACITY)

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def _ingest_data_chunk(self) -> None:
        """
        Read the first 100 rows of ``internal_trades.csv`` and push each trade
        into the C++ engine's SPSC ring-buffer via ``submit_trade()``.

        Column mapping
        ~~~~~~~~~~~~~~
        * ``trade_id``   – hex string  → ``uint64`` via ``int(..., 16)``
        * ``amount``     – float64     → ``double``  (no cast needed)
        * ``timestamp``  – int64 (Unix seconds) → ``uint64`` nanoseconds
                          converted by multiplying by 1_000_000_000

        The third positional argument to ``submit_trade`` is the venue /
        side flag; we hard-code ``1`` (primary venue) as the CSV does not
        carry that field.
        """
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "internal_trades.csv"))
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            trade_id: int = int(str(row["trade_id"]), 16)          # hex string → uint64
            price: int = int(float(row["amount"]))                  # amount to int
            quantity: int = 1                                       # default lot size
            counterparty_id: int = int(row["counterparty_id"])
            timestamp_ns: int = int(row["timestamp"]) * 1_000_000_000  # s → ns

            self.engine.submit_trade(trade_id, price, quantity, counterparty_id, timestamp_ns)


    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> AuditorObservation:
        """
        Reset episode state and return the initial observation.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        # Tick once so the engine has a valid initial timestamp.
        self.engine.tick(time.time_ns())

        # Populate the ring-buffer with real trade data
        self._ingest_data_chunk()

        # Use the new anomaly matrix (likely empty on reset)
        initial_features = self.engine.get_anomaly_matrix().tolist()

        return FinAuditorObservation(
            features=initial_features,
            message="Fin Auditor engine ready.",
            reward=0.0,  # No reward on reset
            done=False
        )

    def step(self, action: AuditorAction) -> AuditorObservation:  # type: ignore[override]
        """
        Advance the engine by one timestep and calculate rewards.
        """
        self._state.step_count += 1

        # 1. Drive the C++ engine with the current wall-clock timestamp.
        self.engine.tick(time.time_ns())

        # 2. Pull the latest ANOMALY matrix out of the engine.
        anomalies: list[list[float]] = self.engine.get_anomaly_matrix().tolist()
        
        # 3. Calculate Asymmetric Cost-Sensitive Reward
        step_reward = 0.0
        
        # We only calculate rewards if there are anomalies and the agent took action
        if anomalies and action and action.decisions:
            # Map decisions to the anomalies
            for i in range(min(len(anomalies), len(action.decisions))):
                decision = action.decisions[i]
                if decision == 2:    # Agent correctly flagged the anomaly
                    step_reward += 1.0
                elif decision == 0:  # Agent passed/ignored a real anomaly!
                    step_reward -= 5.0
                # Action 1 (Investigate) is neutral, 0.0 reward

        return FinAuditorObservation(
            features=anomalies,
            message=f"Processed batch. Found {len(anomalies)} anomalies.",
            reward=step_reward,
            done=False
        )

    # ------------------------------------------------------------------
    # State accessor (required by the OpenEnv interface)
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Return the current episode state (id + step counter)."""
        return self._state
