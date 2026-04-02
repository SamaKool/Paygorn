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
            price: float = float(row["amount"])                     # amount to float
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

        The C++ engine is *not* re-constructed here — re-construction is
        expensive (ring-buffer allocation) and unnecessary between episodes.
        The engine state naturally resets once the buffer wraps around.

        Returns:
            AuditorObservation with an empty feature matrix and a ready message.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        # Tick once so the engine has a valid initial timestamp.
        self.engine.tick(time.time_ns())

        # Populate the ring-buffer with real trade data before returning
        # the first observation so the agent has meaningful features from
        # the very first step.
        self._ingest_data_chunk()

        return FinAuditorObservation(
            features=self.engine.get_observation_matrix().tolist(),
            message="Fin Auditor engine ready.",
        )

    def step(self, action: AuditorAction) -> AuditorObservation:  # type: ignore[override]
        """
        Advance the engine by one timestep.

        The C++ engine ingests the current wall-clock time (nanoseconds),
        updates its internal SPSC ring-buffer and timer-wheel, then exposes
        the latest reconciled feature matrix for the RL agent to observe.

        Args:
            action: ``AuditorAction`` containing per-anomaly decisions
                    (0 = Pass, 1 = Investigate, 2 = Flag).

        Returns:
            ``AuditorObservation`` whose ``features`` field is the
            ``(batch_size, 4)`` matrix produced by ``get_observation_matrix()``.
        """
        self._state.step_count += 1

        # 1. Drive the C++ engine with the current wall-clock timestamp.
        self.engine.tick(time.time_ns())

        # 2. Pull the latest feature matrix out of the engine.
        #    get_observation_matrix() returns a nanobind ndarray; .tolist()
        #    converts it to the nested list expected by AuditorObservation.
        features: list[list[float]] = self.engine.get_observation_matrix().tolist()

        return FinAuditorObservation(
            features=features,
            message="Batch processed",
        )

    # ------------------------------------------------------------------
    # State accessor (required by the OpenEnv interface)
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Return the current episode state (id + step counter)."""
        return self._state
