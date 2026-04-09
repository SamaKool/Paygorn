---
title: Paygorn | HFT Auditor
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

                                          ░██████████           ░██████████            ░██████   
                                          ░██                       ░██               ░██  ░██  
                                          ░██                       ░██              ░██         
                                          ░█████████                ░██               ░████████  
                                          ░██                       ░██                      ░██ 
                                          ░██                       ░██               ░██  ░██  
                                          ░██████████    ░██        ░██       ░██      ░██████   
                                                       
                                                       
                                                       

<div align="center">

**Elite Trade Sentry: A High-Frequency Trade Reconciliation Engine**

*A C++20 native RL environment for microsecond-scale financial anomaly detection*

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-00FF9D?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0tMSAxNEg5VjhIMXYyaDR2NHoiLz48L3N2Zz4=)](https://github.com/meta-pytorch/OpenEnv)
[![C++20](https://img.shields.io/badge/C++20-Engine-2F81F7?style=for-the-badge&logo=cplusplus)](https://en.cppreference.com/w/cpp/20)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-FFB86C?style=for-the-badge&logo=python)](https://python.org)
[![nanobind](https://img.shields.io/badge/nanobind-zero--copy-F85149?style=for-the-badge)](https://github.com/wjakob/nanobind)
[![PPO](https://img.shields.io/badge/PPO-Stable--Baselines3-8B5CF6?style=for-the-badge)](https://stable-baselines3.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-0DB7ED?style=for-the-badge&logo=docker)](https://docker.com)

</div>

---

## 🔍 What is Elite Trade Sentry?

Elite Trade Sentry (ETS) is a production-grade **reinforcement learning environment** that simulates the compliance enforcement pipeline of a high-frequency trading desk. It challenges an agent to detect and classify financial anomalies — spoofed orders, price mismatches, expired receipts — in microsecond-scale trade streams.

The core engine is written entirely in **C++20** and exposed to Python via **[nanobind](https://github.com/wjakob/nanobind)** with zero-copy memory semantics. The RL loop, training harness, and FastAPI server are all Python, but the computation that matters happens at C++ speed.

This is not a simulation that *pretends* to be fast. It's a genuine lock-free, cache-friendly data structure running inside your Python process.

---

## 🏗️ Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│  PYTHON LAYER (OpenEnv / FastAPI / Stable-Baselines3)                │
│                                                                      │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐  │
│  │  train.py   │    │  server/app.py   │    │   inference.py     │  │
│  │  PPO Agent  │    │  FastAPI Server  │    │   LLM Baseline     │  │
│  └──────┬──────┘    └────────┬─────────┘    └─────────┬──────────┘  │
│         └───────────────────┼──────────────────────────┘            │
│                             ▼                                        │
│            ┌─────────────────────────────────┐                       │
│            │   FinAuditorEnvironment          │                       │
│            │   server/fin_auditor_environment │                       │
│            └─────────────┬───────────────────┘                       │
│                          │ nanobind (zero-copy)                      │
├──────────────────────────┼───────────────────────────────────────────┤
│  C++20 NATIVE ENGINE     │                                           │
│                          ▼                                           │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │               ReconciliationEngine                              │ │
│  │                                                                 │ │
│  │  ┌──────────────┐  SPSC  ┌──────────────┐  ┌──────────────┐  │ │
│  │  │ SPSCRingBuffer│ ────→  │  OrderPool   │  │  TimerWheel  │  │ │
│  │  │ Lock-free     │       │  O(1) insert │  │  O(1) expire │  │ │
│  │  │ 2M slots      │       │  Flat array  │  │  3-level     │  │ │
│  │  └──────────────┘       └──────────────┘  └──────────────┘  │ │
│  │                                                                 │ │
│  │       tick() → drain → expire → get_anomaly_matrix()           │ │
│  │       compute_reward(agent_actions) → float                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
````

-----

## ⚡ Key Technical Features

### C++20 Engine Core

| Component | Implementation | Complexity |
|---|---|---|
| **OrderPool** | Flat array with power-of-2 indexing | O(1) insert, O(1) lookup |
| **TimerWheel** | 3-level hierarchical slot structure | O(1) schedule, O(1) expire |
| **SPSCRingBuffer** | Lock-free single-producer/consumer | O(1) push/pop, no mutex |
| **ReconciliationEngine** | Orchestrator for all subsystems | O(N) per tick (N = active) |

### Zero-Copy Python Bridge

The observation matrix — a `(N, 4)` float32 array of trade feature vectors — is returned directly from C++ memory to Python as a `numpy.ndarray`. There is no serialization, no copy, no allocation. Python's garbage collector keeps the engine alive while the array exists, enforced via `nanobind`'s `reference_internal` return policy.

### The MDP (Markov Decision Process)

```text
State (Observation)     Action              Reward
───────────────────     ──────              ──────
Matrix (N, 4):          Per-trade:          Asymmetric:
  time_elapsed          0 = PASS            +1.0  True Positive  (TP)
  price_delta           1 = FLAG            +0.5  True Negative  (TN)
  missing_frequency                         -0.1  False Positive (FP)
  risk_score                                 0.0  False Negative (FN)
```

**Why asymmetric?** Missing a real anomaly (FN) in HFT has catastrophic systemic consequences — a zero reward acts as a strong penalty signal. Falsely flagging a valid trade (FP) is a minor operational cost, so it incurs a small deduction.

-----

## 🎯 Task Difficulty Modes

The environment supports three difficulty levels, controlled via the `TASK_ID` environment variable or the `/config/difficulty` API endpoint:

| Mode | Anomaly Rate | Signal | Counterparty Clusters |
|---|---|---|---|
| **EASY** | 100% anomalies | Deterministic; `risk_score > 0.5` always anomalous | High-risk: [70–99] |
| **MEDIUM** | 50% anomalies | Probabilistic; 80% high-risk = anomaly, 10% FP chance | High-risk: [70–99], Low-risk: [0–19] |
| **HARD** | 20% anomalies | Adversarial; weak correlations, high noise | Mixed — no clean separation |

-----

## 🚀 Quick Start

### Prerequisites

  - Python 3.12 (via `uv`)
  - C++ compiler: MSVC 2022 (Windows) or GCC/Clang (Linux)
  - CMake ≥ 3.15

### 1\. Clone & Install

```bash
git clone [https://github.com/SamaKool/Paygorn.git](https://github.com/SamaKool/Paygorn.git)
cd Paygorn
pip install uv
uv sync
```

### 2\. Build the C++ Engine

```bash
# Windows (Visual Studio 2022)
python build_engine.py

# Linux/macOS
python build_engine.py

# Memory-safe build for constrained environments (Docker/HF Spaces)
python build_engine.py --docker-safe
```

On success you will see:

```text
✓  Build complete.  Extension available at: D:\Paygorn\hft_auditor.cp312-win_amd64.pyd
```

### 3\. Run the Server

```bash
cd server
uv run python app.py
# → http://localhost:8000
```

### 4\. Train the PPO Agent

```bash
uv run python train.py
# Checkpoints saved to ./logs/ every 5,000 steps
```

### 5\. Run LLM Inference Baseline

```bash
export HF_TOKEN=hf_...
uv run python inference.py
```

-----

## 🖥️ Command Center Dashboard

The FastAPI server serves an interactive HFT Command Center at `http://localhost:8000/`. It provides:

  - **Real-time telemetry**: engine latency (μs), audit accuracy, throughput (M/s), buffer saturation
  - **Authority Checklist**: live status of the C++ binary, API key, model, and LLM connection
  - **LLM Router Config**: inject HuggingFace / OpenAI / Anthropic keys and auto-discover models
  - **Manual Override**: trigger `reset`, `step_optimal`, and `step_random` actions from the browser
  - **Execution Ledger**: live log of each step's reward, TP, FP, and done state
  - **Classification Pulse**: animated bar chart of TP/TN/FP/FN from the last step
  - **HARD Mode CRT Effect**: the UI applies an adversarial visual overlay when difficulty = HARD

-----

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | GET | Interactive HFT Command Center UI |
| `GET /state` | GET | Single source of truth: all telemetry + health |
| `POST /api/reset` | POST | Flush SPSC buffer and reset episode |
| `POST /api/step` | POST | Execute one agent step (`random`, `perfect`, `llm`, `ppo`) |
| `POST /config/llm` | POST | Validate API key and discover provider models |
| `POST /config/default` | POST | Zero-config mode using `HF_TOKEN` env var |
| `POST /config/difficulty` | POST | Switch difficulty (`EASY`, `MEDIUM`, `HARD_ADVERSARIAL`) |
| `GET /docs` | GET | Full OpenAPI / Swagger documentation |

### Example: Run a step with a random agent

```bash
curl -X POST http://localhost:8000/api/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "random"}'
```

```json
{
  "status": "success",
  "step": 1,
  "reward": 0.3125,
  "done": false,
  "tp": 8,
  "tn": 12,
  "fp": 3,
  "fn": 17
}
```

-----

## 🧠 Python Integration

### Direct Environment Usage

```python
from server.fin_auditor_environment import FinAuditorEnvironment
from models import AuditorAction
import numpy as np

env = FinAuditorEnvironment()

# Reset — returns an empty feature matrix to flush state
obs = env.reset()

# Step — generate a batch, expire trades, compute reward
action = AuditorAction(decisions=[1] * 40)   # flag all as anomalous
obs = env.step(action)

print(f"Reward:   {obs.reward:.4f}")
print(f"Features: {np.array(obs.features).shape}")  # (N, 4)
print(f"TP/FP:    {env.state.last_tp} / {env.state.last_fp}")
```

### Gymnasium Wrapper (for PPO / SB3)

```python
from train import GymnasiumFinAuditorEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv([lambda: GymnasiumFinAuditorEnv()])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = PPO("MlpPolicy", env, verbose=1, n_steps=256, batch_size=64)
model.learn(total_timesteps=100_000)
model.save("ppo_fin_auditor_final")
```

### LLM Zero-Shot Baseline (via HuggingFace Router)

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="hf_...",
    base_url="[https://router.huggingface.co/v1](https://router.huggingface.co/v1)"
)

# The engine gives the LLM a JSON decision schema
# The LLM outputs {"decisions": [0, 1, 1, 0, ...]} for 40 trades
response = await client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Flag anomalies in this trading batch."}],
    response_format={"type": "json_object"},
    temperature=0.2
)
```

-----

## 🏭 Project Structure

```text
Paygorn/
│
├── hf auditor/                         # C++20 Engine Source
│   ├── CMakeLists.txt                  # Build configuration
│   ├── src/
│   │   ├── auditor.cpp                 # nanobind module entry point
│   │   ├── reconciliation_engine.hpp   # Main orchestrator (C++20)
│   │   ├── order_pool.hpp              # O(1) flat-array trade storage
│   │   ├── spsc_ring_buffer.hpp        # Lock-free SPSC queue
│   │   └── timer_wheel.hpp             # O(1) hierarchical expiration
│   └── benchmarks/
│       └── benchmark_engine.cpp        # Google Benchmark harness
│
├── server/                             # Python API Layer
│   ├── app.py                          # FastAPI server + Command Center UI
│   ├── fin_auditor_environment.py      # OpenEnv environment wrapper
│   └── __init__.py
│
├── build_engine.py                     # Cross-platform CMake build script
├── train.py                            # PPO training loop (Stable-Baselines3)
├── inference.py                        # LLM zero-shot inference baseline
├── final_check.py                      # End-to-end validation harness
├── models.py                           # Pydantic Action / Observation types
├── openenv.yaml                        # OpenEnv task manifest
├── pyproject.toml                      # Python project metadata
├── Dockerfile                          # Multi-stage Docker build
└── README.md                           # This file
```

-----

## 🐳 Docker & Hugging Face Spaces

The Dockerfile uses a 2-stage build strategy for memory-constrained environments:

1.  **Stage 1 (builder)**: Installs Python deps, then checks for a pre-compiled `.so`. If found, it uses it directly. If not, it compiles with `-j1 -O1` (`--docker-safe`) to cap RAM usage at \~1.2 GB vs \~5 GB for a full `-O3` build.

2.  **Stage 2 (runtime)**: A minimal image containing only the venv, app code, and the compiled `.so`.

<!-- end list -->

```bash
# Local build
docker build -t paygorn:latest .
docker run -p 8000:7860 paygorn:latest

# Deploy to Hugging Face Spaces
openenv push --repo-id your-username/paygorn
```

### Recommended Workflow for HF Spaces

Pre-compile the `.so` locally before pushing:

```bash
python build_engine.py          # Produces hft_auditor.cpXXX-linux_x86_64.so
git add hft_auditor*.so
git commit -m "Add pre-compiled engine binary"
openenv push
```

The Docker builder will detect the pre-compiled binary and skip the expensive CMake step entirely.

-----

## 🔧 Build Options

```bash
# Standard Windows build (Visual Studio 17 2022)
python build_engine.py

# Standard Linux build (Ninja / Make)
python build_engine.py

# Memory-safe build (-O1, single threaded): for HF Spaces / CI
python build_engine.py --docker-safe

# Rebuild from scratch (cleans build/ directory first)
python build_engine.py   # always cleans first
```

-----

## 📊 Reward Function Design

The reward function encodes the asymmetric cost structure of real HFT compliance:

```text
R(action, truth) = {
    +1.0   if action=FLAG   AND truth=ANOMALY   → True  Positive (TP)
    +0.5   if action=PASS   AND truth=SAFE      → True  Negative (TN)
    -0.1   if action=FLAG   AND truth=SAFE      → False Positive (FP) 
     0.0   if action=PASS   AND truth=ANOMALY   → False Negative (FN)
}
```

**Why zero for FN?** A zero reward means the agent gains nothing from missing a real anomaly, which acts as an implicit penalty when normalized against the opportunity cost of the missed TP. Combined with the asymmetric discount factor in PPO's GAE, this creates strong pressure to avoid FNs without requiring a separate penalty term.

-----

## 🤝 OpenEnv Compliance

ETS is fully compliant with the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) standard. The environment exposes:

  - `reset() → AuditorObservation` — initializes and returns a clean state
  - `step(AuditorAction) → AuditorObservation` — processes one decision batch
  - `SUPPORTS_CONCURRENT_SESSIONS = True` — multiple WebSocket clients can connect simultaneously
  - Three registered tasks in `openenv.yaml`: `anomaly_detection_easy`, `anomaly_detection_medium`, `anomaly_detection_hard`

-----

\<div align="center"\>

Built with ⚡ C++20, 🐍 Python 3.12, and 🧠 RL by the PayGorn team.

\</div\>

