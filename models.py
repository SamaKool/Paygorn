# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
from pydantic import StrictFloat

# FIX: Import the OpenEnv base classes instead of BaseModel
from openenv.core.env_server import Action, Observation

class AuditorAction(Action):
    # Decisions for each anomaly in the batch: 0=Pass, 1=Investigate, 2=Flag
    decisions: List[int]

class AuditorObservation(Observation):
    # This is the (batch_size, 4) matrix Samarth will send
    # Features: [time_elapsed, price_delta, missing_freq, risk_score]
    features: List[List[StrictFloat]] 
    message: str = "Batch processed"