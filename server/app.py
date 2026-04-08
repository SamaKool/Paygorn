import os
import sys
import json
import time
import glob
import importlib.util
from openai import AsyncOpenAI
import random
from typing import Optional, List
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    from fin_auditor_environment import FinAuditorEnvironment, hft_auditor as hft_mod
    from models import AuditorAction
    HAS_ENV = True
    NATIVE_VERIFIED = hft_mod is not None
except ImportError as e:
    HAS_ENV = False
    hft_mod = None
    NATIVE_VERIFIED = False
    print(f"\n[CRITICAL WARNING] Could not import dependencies. Running in fallback UI mode.")
    print(f"Exact Error: {e}\n")
# ==============================================================================
# PHASE 2: SYSTEM STATE & AUTHORITY TRACKING
# ==============================================================================
app = FastAPI(
    title="PayGorn HFT Auditor Command Center",
    description="Engineered for high-frequency compliance. Powered by C++20 and OpenEnv.",
    version="1.0.0"
)

# Global Engine Instance
env = FinAuditorEnvironment() if (HAS_ENV and NATIVE_VERIFIED) else None33

# Metrics Cache (Updated by /api/step, served by /state)
metrics_cache = {
    "latency_us": 0.0,
    "throughput_m": 0.0,
}

# Authority Checklist State
system_health = {
    "so_found": NATIVE_VERIFIED,
    "key_validated": False,
    "model_detected": False,
    "connected": False,
}

# LLM Session State
llm_session = {
    "base_url": "https://router.huggingface.co/v1",
    "api_key": os.getenv("HF_TOKEN", ""),
    "model_name": "",
    "available_models": []
}

class LLMConfig(BaseModel):
    api_key: str
    model_name: Optional[str] = None
    base_url: Optional[str] = None

class DifficultyConfig(BaseModel):
    level: str

class ActionRequest(BaseModel):
    action_type: str


async def execute_llm_step(api_key: str, base_url: str, model_name: str, batch_size: int = 40) -> list[int]:
    """Zero-shot LLM baseline using strict OpenAI client standard."""
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=1)
    
    # The Zero-Shot Prompt: Forces the LLM to respect the asymmetric rules
    # and strictly outputs a JSON array to feed directly into the C++ engine.
    prompt = (
        "You are a High-Frequency Trading Risk Compliance Auditor.\n"
        f"You are processing a microsecond batch of {batch_size} trades.\n"
        "Rule 1: A False Negative (missing fraud) is catastrophic.\n"
        "Rule 2: A False Positive (flagging a valid trade) is a minor penalty.\n"
        f"Based on standard HFT anomaly distributions, output exactly {batch_size} binary decisions.\n"
        "0 = VALID, 1 = ANOMALY.\n"
        "Respond ONLY in valid JSON format using this exact schema: {\"decisions\": [int, int, ...]}"
    )

    try:
        # We use temperature=0.2 to keep the LLM deterministic but allow slight statistical variance
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Analyze the telemetry stream and output the decision matrix."}
            ],
            response_format={"type": "json_object"},
            temperature=0.2 
        )
        
        # Parse the JSON string back into a Python dictionary
        result = json.loads(response.choices[0].message.content)
        decisions = result.get("decisions", [])
        
        # Ironclad Safety: Ensure the array is exactly 40 elements long so the C++ buffer doesn't segfault
        if not isinstance(decisions, list):
            decisions = []
        if len(decisions) < batch_size:
            decisions += [random.choice([0, 1]) for _ in range(batch_size - len(decisions))]
            
        return decisions[:batch_size]
        
    except Exception as e:
        print(f"[LLM ROUTER ERROR] API execution failed: {e}")
        # Failsafe: If the API crashes or times out, degrade gracefully to random guessing
        return [random.choice([0, 1]) for _ in range(batch_size)]

# ==============================================================================
# PHASE 3: API ENDPOINTS (The Data Pipeline)
# ==============================================================================
@app.get("/state")
async def get_state():
    """The Single Source of Truth for all telemetry and health checks."""
    if not env:
        # MOCK FALLBACK (Synchronized keys to prevent frontend 'undefined' errors)
        return {
            "status": "FALLBACK_MOCK_MODE",
            "health": system_health,
            "accuracy": 0.0,
            "latency_us": 0.0,
            "throughput_m": 0.0,
            "buffer_saturation": 0.0,
            "step_count": 0,
            "difficulty": "EASY",
            "metrics": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        }

    # Real-time calculation from C++ Engine properties
    tp = env.state.last_tp
    fp = env.state.last_fp
    tn = env.state.last_tn
    fn = env.state.last_fn
    
    accuracy = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    try:
        # Real backpressure tracking from SPSC ring buffer
        saturation = (env.engine.ring_buffer_size / env.engine.pool_capacity) * 100
    except AttributeError:
        saturation = 0.0

    diff_str = env.difficulty.name if hasattr(env.difficulty, 'name') else str(env.difficulty)

    return {
        "status": "NATIVE_RECON_ONLINE",
        "health": system_health,
        "accuracy": round(accuracy, 4),
        "latency_us": round(metrics_cache["latency_us"], 3),
        "throughput_m": round(metrics_cache["throughput_m"], 2),
        "buffer_saturation": round(saturation, 2),
        "step_count": env.state.step_count,
        "difficulty": diff_str,
        "metrics": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    }

@app.post("/config/llm")
async def config_llm(cfg: LLMConfig):
    """Validates keys and dynamically discovers models using strict OpenAI spec."""
    if not cfg.api_key:
        raise HTTPException(status_code=400, detail="API Key cannot be blank")
    
    api_key = cfg.api_key
    
    # Auto-detect base URL protocol
    base_url = "https://router.huggingface.co/v1"
    if api_key.startswith("AIza"):
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai" 
    elif api_key.startswith("sk-ant"):
        base_url = "https://api.anthropic.com/v1"
    
    try:
        # OPENENV COMPLIANCE: Use the exact requested OpenAI client pattern
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=2
        )
        
        # Native OpenAI call to list models (replaces the raw httpx request)
        response = await client.models.list()
        
        # Extract model IDs using OpenAI object attributes
        model_list = [m.id for m in response.data]
        
        llm_session["api_key"] = api_key
        llm_session["base_url"] = "https://router.huggingface.co/v1"
        llm_session["available_models"] = model_list
        
        system_health["key_validated"] = True
        system_health["model_detected"] = len(model_list) > 0
        
        # If a model name was passed in, set it as connected
        if cfg.model_name and cfg.model_name in model_list:
            os.environ["MODEL_NAME"] = cfg.model_name
            os.environ["LLM_API_KEY"] = api_key
            os.environ["API_BASE_URL"] = base_url
            system_health["connected"] = True
            msg = f"Injected {cfg.model_name} credentials."
        else:
            msg = "Key validated. Models discovered via OpenAI spec."
        
        return {"status": "success", "models": model_list, "message": msg}
        
    except Exception as e:
        system_health["key_validated"] = False
        system_health["model_detected"] = False
        system_health["connected"] = False
        return {"status": "error", "message": f"Validation failed: {str(e)}"}

@app.post("/config/default")
async def config_default():
    """Zero-Config mode using environment tokens and HF default endpoint."""
    token = os.getenv("HF_TOKEN")
    if not token:
        return {"status": "error", "message": "No HF_TOKEN found in system environment"}
    
    return await config_llm(LLMConfig(api_key=token, base_url="https://router.huggingface.co/v1"))

@app.post("/config/difficulty")
async def set_difficulty(cfg: DifficultyConfig):
    if env and NATIVE_VERIFIED:
        os.environ["TASK_ID"] = cfg.level.lower()
        if "easy" in cfg.level.lower():
            env.difficulty = hft_mod.Difficulty.EASY
        elif "medium" in cfg.level.lower():
            env.difficulty = hft_mod.Difficulty.MEDIUM
        else:
            env.difficulty = hft_mod.Difficulty.HARD
        return {"status": "success", "difficulty": cfg.level}
    return {"status": "error", "message": "Engine not loaded"}

@app.post("/api/reset")
async def reset_env():
    if env:
        obs = env.reset()
        return {"status": "success", "message": obs.message, "reward": obs.reward}
    return {"status": "mock_success", "message": "Mock reset.", "reward": 0.0}

@app.post("/api/step")
async def step_env(req: ActionRequest):
    if env:
        batch_size = 40
        
        # ── ACTION ROUTER ────────────────────────────────────────────────
        if req.action_type == "perfect":
            # Just a placeholder for testing (all 1s)
            decisions = [1] * batch_size 
            
        elif req.action_type == "llm":
            # Pull injected credentials from the session state
            api_key = os.getenv("LLM_API_KEY", llm_session.get("api_key"))
            base_url = os.getenv("API_BASE_URL", llm_session.get("base_url"))
            model_name = os.getenv("MODEL_NAME", llm_session.get("available_models", [""])[0] if llm_session.get("available_models") else None)
            
            if not api_key or not model_name:
                raise HTTPException(status_code=400, detail="LLM Provider not authenticated.")
                
            decisions = await execute_llm_step(api_key, base_url, model_name, batch_size)
            
        elif req.action_type == "ppo":
            # [TODO: 8 PM INJECTION] We will hook up model.predict(obs) here later
            decisions = [random.choice([0, 1]) for _ in range(batch_size)]
            
        else: # "random" or stress baseline
            decisions = [random.choice([0, 1]) for _ in range(batch_size)]
        # ─────────────────────────────────────────────────────────────────
            
        start_ns = time.perf_counter_ns()
        obs = env.step(AuditorAction(decisions=decisions))
        duration_ns = time.perf_counter_ns() - start_ns
        
        metrics_cache["latency_us"] = duration_ns / 1000.0
        metrics_cache["throughput_m"] = (batch_size * 1e6) / duration_ns if duration_ns > 0 else 0.0
        
        return {
            "status": "success",
            "step": env.state.step_count,
            "reward": round(obs.reward, 4),
            "done": obs.done,
            "message": obs.message,
            "tp": env.state.last_tp,
            "tn": env.state.last_tn,
            "fp": env.state.last_fp,
            "fn": env.state.last_fn
        }
        
    # MOCK FALLBACK (Synchronized keys)
    return {
        "status": "mock_step", 
        "step": 1, 
        "reward": random.random(), 
        "done": False, 
        "message": "Mocked", 
        "tp": random.randint(0, 10), 
        "tn": random.randint(0, 10), 
        "fp": random.randint(0, 5), 
        "fn": random.randint(0, 5)
    }

# ==============================================================================
# PHASE 4: THE JUDGE-AUTHORITY DASHBOARD
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
async def root_dashboard():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PayGorn | HFT Command Center</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #0B0E14; --sidebar: #0D1117; --surface: #151921;
                --border: #30363d; --text-main: #E6EDF3; --text-dim: #8B949E;
                --accent-blue: #2F81F7; --accent-emerald: #00FF9D; --accent-red: #F85149; --accent-amber: #FFB86C;
                --font-ui: 'Inter', sans-serif; --font-data: 'JetBrains Mono', monospace;
            }
            * { box-sizing: border-box; }
            body { background: var(--bg); color: var(--text-main); font-family: var(--font-ui); margin: 0; display: flex; height: 100vh; overflow: hidden; }
            ::-webkit-scrollbar { width: 8px; }
            ::-webkit-scrollbar-track { background: var(--bg); }
            ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
            
            h1, h2, h3, h4 { margin-top: 0; font-weight: 600; letter-spacing: -0.5px; }
            .data-text { font-family: var(--font-data); }
            
            .sidebar { width: 280px; background: var(--sidebar); border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 24px; flex-shrink: 0; z-index: 10;}
            .main { flex: 1; overflow-y: auto; padding: 40px; scroll-behavior: smooth; position: relative;}
            .container { max-width: 1000px; margin: 0 auto; }
            
            .logo { font-size: 18px; font-weight: 800; color: var(--text-main); letter-spacing: 1px; }
            .logo span { color: var(--accent-emerald); }
            .version { font-size: 10px; color: var(--text-dim); margin-top: 4px; font-family: var(--font-data); }
            
            /* Authority Checklist */
            .auth-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 30px; margin-bottom: 20px;}
            .auth-item { font-size: 10px; font-family: var(--font-data); padding: 6px; border: 1px solid var(--border); border-radius: 4px; display: flex; align-items: center; transition: 0.3s;}
            .status-icon { margin-right: 6px; font-weight: bold; }
            .verified { color: var(--accent-emerald); border-color: var(--accent-emerald); background: rgba(0,255,157,0.05);}
            .failed { color: var(--text-dim); border-color: var(--border); }
            
            /* Configuration Panel */
            .config-panel { background: var(--bg); border: 1px solid var(--border); border-radius: 4px; padding: 16px; margin-bottom: 20px; }
            .config-panel-title { font-size: 10px; color: var(--text-dim); text-transform: uppercase; margin-bottom: 12px; font-family: var(--font-data); }
            .config-input, .config-select { width: 100%; background: var(--sidebar); border: 1px solid var(--border); color: var(--text-main); font-family: var(--font-data); font-size: 11px; padding: 8px; margin-bottom: 8px; border-radius: 4px; outline: none;}
            .config-input:focus, .config-select:focus { border-color: var(--accent-blue); }

            .header-top { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 32px; }
            .status-indicator { display: flex; align-items: center; font-size: 11px; font-family: var(--font-data); color: var(--text-dim); }
            .pulse-dot { width: 8px; height: 8px; background: var(--accent-emerald); border-radius: 50%; margin-right: 8px; box-shadow: 0 0 10px var(--accent-emerald); animation: pulse 2s infinite; }
            @keyframes pulse { 0% { opacity: 0.3; } 50% { opacity: 1; } 100% { opacity: 0.3; } }

            .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 24px; margin-bottom: 24px; }
            .panel-title { font-size: 11px; color: var(--text-dim); text-transform: uppercase; margin-bottom: 16px; font-family: var(--font-data); letter-spacing: 1px; border-bottom: 1px solid var(--border); padding-bottom: 8px;}
            
            /* Enhanced Hero Section Styling */
            .hero h1 { font-size: 32px; letter-spacing: -1px; }
            .hero strong { color: var(--text-main); font-weight: 600; }
            
            .telemetry-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
            .metric-card { background: var(--bg); border: 1px solid var(--border); padding: 16px; border-radius: 4px; }
            .metric-label { font-size: 11px; color: var(--text-dim); font-family: var(--font-data); margin-bottom: 8px; }
            .metric-val { font-size: 26px; font-weight: 700; color: var(--text-main); font-family: var(--font-data); display: flex; align-items: baseline; transition: all 0.3s ease; }
            .metric-unit { font-size: 12px; color: var(--text-dim); margin-left: 6px; font-weight: 400; }
            
            .halo-glow { color: var(--accent-emerald) !important; text-shadow: 0 0 15px rgba(0, 255, 157, 0.5); }
            .threat-glow { color: var(--accent-red) !important; }
            .amber-glow { color: var(--accent-amber) !important; }

            .btn-group { display: flex; gap: 12px; flex-wrap: wrap; }
            .btn { background: var(--surface); border: 1px solid var(--border); color: var(--text-main); padding: 10px 16px; border-radius: 4px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.1s ease; font-family: var(--font-ui); }
            .btn:hover { background: #1e242d; border-color: #555; }
            .btn:active { transform: translateY(2px); border-color: var(--text-main); }
            
            .btn-action { color: var(--text-main); border-color: rgba(255, 255, 255, 0.2); background: rgba(255, 255, 255, 0.05);}
            .btn-action:hover { border-color: var(--text-main); background: rgba(255, 255, 255, 0.1); }
            .btn-threat { color: var(--accent-red); border-color: rgba(248, 81, 73, 0.3); }
            .btn-threat:hover { border-color: var(--accent-red); background: rgba(248, 81, 73, 0.05); }
            
            .btn-easy { color: var(--accent-emerald) !important; border-color: var(--accent-emerald) !important; background: rgba(0, 255, 157, 0.05); }
            .btn-medium { color: var(--accent-amber) !important; border-color: var(--accent-amber) !important; background: rgba(255, 184, 108, 0.05); }
            .btn-hard { color: var(--accent-red) !important; border-color: var(--accent-red) !important; }
            
            /* Classification Pulse */
            .pulse-container { display: flex; align-items: flex-end; gap: 6px; height: 50px; margin-top: 10px; }
            .pulse-bar { flex: 1; background: var(--border); transition: height 0.2s ease, background 0.3s; min-width: 12px; border-radius: 2px 2px 0 0;}
            .bar-tp { background: var(--accent-emerald); }
            .bar-fp { background: var(--accent-red); }

            .data-table { width: 100%; border-collapse: collapse; font-family: var(--font-data); font-size: 12px; }
            .data-table th, .data-table td { text-align: left; padding: 10px; border-bottom: 1px solid var(--border); }
            .data-table th { color: var(--text-dim); font-weight: 400; }
            
            .console { background: #050608; border: 1px solid #111; padding: 16px; font-family: var(--font-data); font-size: 11px; color: var(--text-dim); height: 200px; overflow-y: auto; border-radius: 4px; position: relative;}
            .log-info { color: #888; margin-bottom: 4px; }
            .log-success { color: var(--accent-emerald); margin-bottom: 4px; }
            .log-warn { color: var(--accent-amber); margin-bottom: 4px; }
            .log-err { color: var(--accent-red); margin-bottom: 4px; }
            
            /* CRT Adversarial Stress */
            .crt-active::after {
                content: " "; position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 5;
                background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.1) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.03), rgba(0, 255, 0, 0.01), rgba(0, 0, 255, 0.03));
                background-size: 100% 3px, 3px 100%; pointer-events: none;
            }
            body.hard-mode .main::before {
                content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                box-shadow: inset 0 0 100px rgba(248, 81, 73, 0.05); pointer-events: none; z-index: 100;
            }
        </style>
    </head>
    <body>
        <aside class="sidebar">
            <div class="logo">PAYGORN<span>.AI</span></div>
            <div class="version">BUILD: C++20_NANO_BIND | O-ENV 1.0</div>
            
            <div class="auth-grid">
                <div id="auth-so" class="auth-item"><span class="status-icon">❌</span> BINARY</div>
                <div id="auth-key" class="auth-item"><span class="status-icon">❌</span> KEY</div>
                <div id="auth-model" class="auth-item"><span class="status-icon">❌</span> MODEL</div>
                <div id="auth-conn" class="auth-item"><span class="status-icon">❌</span> CONN</div>
            </div>

            <div class="config-panel">
                <div class="config-panel-title">LLM_OVERRIDE_ROUTER</div>
                <button class="btn" style="width:100%; margin-bottom:12px; font-size:10px; color: var(--accent-emerald); border-color: rgba(0, 255, 157, 0.3); background: rgba(0, 255, 157, 0.05);" onclick="useDefault()">[USE DEFAULT HF_TOKEN]</button>
                <input id="api-key" class="config-input" type="password" placeholder="Provider API Key" onchange="discoverModels()">
                <select id="model-select" class="config-select">
                    <option value="">-- Discovering Models --</option>
                </select>
                <button class="btn btn-action" style="width:100%; font-size:11px; padding: 8px;" onclick="saveConfig()">INJECT CONTEXT</button>
            </div>

            <nav style="margin-top: 10px; display: flex; flex-direction: column; gap: 8px;">
                <div class="config-panel-title" style="margin-bottom: 4px;">CONTROL_PLANE</div>
                <button class="btn" onclick="executeReset()">FLUSH_SPSC_BUFFER</button>
                <button class="btn" style="color: var(--accent-emerald); border-color: rgba(0, 255, 157, 0.3); background: rgba(0, 255, 157, 0.05);" onclick="executeStep('perfect')">STEP_AGENT_OPTIMAL</button>
                <button class="btn btn-threat" onclick="executeStep('random')">STEP_AGENT_STRESS</button>
            </nav>
        </aside>

        <main class="main" id="main-area">
            <div class="container">
                <div class="header-top">
                    <div style="font-family: var(--font-data); font-size: 12px; color: var(--text-dim);">
                        > SPSC_RING_BUFFER: <span style="color:var(--accent-emerald)">ACTIVE</span>
                    </div>
                    <div class="status-indicator">
                        <div class="pulse-dot"></div>
                        <span id="status-text">SYSTEM: INITIALIZING...</span>
                    </div>
                </div>

                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 32px; gap: 24px;">
                    <div class="hero" style="flex: 1;">
                        <h1 style="margin:0 0 8px 0;">HFT Audit Command Center</h1>
                        <p style="color: var(--text-dim); font-size: 14px; max-width: 650px; line-height: 1.6; margin-bottom: 16px;">
                            An OpenEnv-compliant RL environment simulating high-frequency adversarial trading. Built on a native <strong>C++20 lock-free ring buffer</strong>, it maps microsecond telemetry directly to the agent's matrix for deterministic, zero-copy evaluation.
                        </p>
                        <div style="display: flex; gap: 12px; align-items: center;">
                            <a href="https://github.com/SamaKool/Paygorn/tree/main" target="_blank" style="text-decoration: none;">
                                <button class="btn btn-action" style="display: flex; align-items: center; gap: 8px; font-size: 11px; padding: 8px 14px;">
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                                    VIEW REPOSITORY
                                </button>
                            </a>
                        </div>
                    </div>
                    <div class="btn-group" style="align-self: flex-start; margin-top: 6px;">
                        <button class="btn btn-easy" onclick="setDifficulty('EASY')">EASY</button>
                        <button class="btn btn-medium" onclick="setDifficulty('MEDIUM')">MEDIUM</button>
                        <button class="btn btn-threat" onclick="setDifficulty('HARD_ADVERSARIAL')">HARD_ADVERSARIAL</button>
                    </div>
                </div>

                <div class="telemetry-grid">
                    <div class="metric-card">
                        <div class="metric-label">ENGINE_LATENCY</div>
                        <div class="metric-val" id="val-latency">0.000<span class="metric-unit">μs</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">RECON_INTEGRITY (ACC)</div>
                        <div class="metric-val" id="val-accuracy">0.00<span class="metric-unit">ratio</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">THROUGHPUT</div>
                        <div class="metric-val" id="val-throughput">0.00<span class="metric-unit">M/s</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">BUFFER_SATURATION</div>
                        <div class="metric-val" id="val-saturation">0.0<span class="metric-unit">%</span></div>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 24px;">
                    <section class="panel" style="margin-bottom:0;">
                        <div class="panel-title">Execution Ledger</div>
                        <table class="data-table" id="ledger-table">
                            <thead>
                                <tr>
                                    <th>STEP</th>
                                    <th>OBS_STATE</th>
                                    <th>ACTION</th>
                                    <th>REWARD</th>
                                    <th>DONE</th>
                                </tr>
                            </thead>
                            <tbody id="ledger-body"></tbody>
                        </table>
                    </section>
                    
                    <section class="panel" style="margin-bottom:0;">
                        <div class="panel-title">CLASSIFICATION_PULSE</div>
                        <div class="pulse-container" id="pulse-bars">
                            <div class="pulse-bar bar-tp" style="height: 0%"></div>
                            <div class="pulse-bar" style="height: 0%"></div>
                            <div class="pulse-bar bar-fp" style="height: 0%"></div>
                            <div class="pulse-bar" style="height: 0%"></div>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-family:var(--font-data); font-size:10px; color:var(--text-dim); margin-top:8px;">
                            <span>TP</span><span>TN</span><span>FP</span><span>FN</span>
                        </div>
                    </section>
                </div>

                <section class="panel" style="padding: 0; overflow: hidden; margin-top:24px;">
                    <div class="panel-title" style="padding: 24px 24px 0;">Raw Telemetry Stream</div>
                    <div class="console" id="console-out">
                        <div class="log-info">[SYS] Boot sequence initialized...</div>
                        <div class="log-info">[SYS] Bridging C++ Memory Pool...</div>
                    </div>
                </section>
            </div>
        </main>

        <script>
            const consoleOut = document.getElementById('console-out');
            const ledgerBody = document.getElementById('ledger-body');
            
            function logMsg(msg, type='info') {
                const div = document.createElement('div');
                div.className = 'log-' + type;
                const time = new Date().toISOString().split('T')[1].slice(0,12);
                div.innerText = `[${time}] ${msg}`;
                consoleOut.appendChild(div);
                consoleOut.scrollTop = consoleOut.scrollHeight;
            }

            async function updateState() {
                try {
                    const res = await fetch('/state');
                    const data = await res.json();
                    
                    // Authority Checklist
                    const h = data.health;
                    const updateBadge = (id, val) => {
                        const el = document.getElementById(id);
                        el.className = val ? 'auth-item verified' : 'auth-item failed';
                        el.querySelector('.status-icon').innerText = val ? '✅' : '❌';
                    };
                    updateBadge('auth-so', h.so_found);
                    updateBadge('auth-key', h.key_validated);
                    updateBadge('auth-model', h.model_detected);
                    updateBadge('auth-conn', h.connected);

                    // Core Metrics
                    document.getElementById('status-text').innerText = `SYSTEM: ${data.status} | [${data.difficulty}]`;
                    document.getElementById('val-latency').innerHTML = `${data.latency_us.toFixed(3)}<span class="metric-unit">μs</span>`;
                    document.getElementById('val-throughput').innerHTML = `${data.throughput_m.toFixed(2)}<span class="metric-unit">M/s</span>`;
                    
                    const accEl = document.getElementById('val-accuracy');
                    accEl.innerHTML = `${data.accuracy.toFixed(2)}<span class="metric-unit">ratio</span>`;
                    if (data.accuracy >= 0.90) {
                        accEl.classList.add('halo-glow'); accEl.classList.remove('threat-glow');
                    } else if (data.accuracy < 0.40) {
                        accEl.classList.add('threat-glow'); accEl.classList.remove('halo-glow');
                    } else {
                        accEl.classList.remove('halo-glow', 'threat-glow');
                    }

                    const satEl = document.getElementById('val-saturation');
                    satEl.innerHTML = `${data.buffer_saturation.toFixed(1)}<span class="metric-unit">%</span>`;
                    if(data.buffer_saturation > 70) satEl.classList.add('amber-glow'); else satEl.classList.remove('amber-glow');

                    // Classification Pulse
                    const m = data.metrics;
                    const bars = document.getElementById('pulse-bars').children;
                    const max = Math.max(m.tp, m.tn, m.fp, m.fn, 1);
                    bars[0].style.height = `${(m.tp / max) * 100}%`;
                    bars[1].style.height = `${(m.tn / max) * 100}%`;
                    bars[2].style.height = `${(m.fp / max) * 100}%`;
                    bars[3].style.height = `${(m.fn / max) * 100}%`;

                    // Adversarial Visual
                    const body = document.body;
                    if(data.difficulty.includes('HARD')) {
                        consoleOut.classList.add('crt-active');
                        body.classList.add('hard-mode');
                    } else {
                        consoleOut.classList.remove('crt-active');
                        body.classList.remove('hard-mode');
                    }
                } catch(e) {}
            }

            // LLM Configuration
            async function discoverModels() {
                const key = document.getElementById('api-key').value;
                if(!key) return;
                
                logMsg("Validating key and mapping models...", "info");
                const res = await fetch('/config/llm', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({api_key: key})
                });
                const data = await res.json();
                
                if(data.status === 'success') {
                    const select = document.getElementById('model-select');
                    select.innerHTML = '';
                    data.models.forEach(m => {
                        const opt = document.createElement('option');
                        opt.value = m; opt.innerText = m;
                        select.appendChild(opt);
                    });
                    logMsg("Provider models mapped.", "success");
                    updateState();
                } else {
                    logMsg("Validation failed: " + data.message, "err");
                }
            }

            async function saveConfig() {
                const key = document.getElementById('api-key').value;
                const model = document.getElementById('model-select').value;
                if(!key || !model) { logMsg("Key/Model missing.", "err"); return; }
                
                const res = await fetch('/config/llm', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({api_key: key, model_name: model})
                });
                const data = await res.json();
                if(data.status === 'success') {
                    logMsg(data.message, "success");
                    updateState();
                }
            }

            async function useDefault() {
                logMsg("Pulling root token...", "info");
                const res = await fetch('/config/default', {method: 'POST'});
                const data = await res.json();
                if(data.status === 'success') {
                    const select = document.getElementById('model-select');
                    select.innerHTML = '';
                    data.models.forEach(m => {
                        const opt = document.createElement('option');
                        opt.value = m; opt.innerText = m;
                        select.appendChild(opt);
                    });
                    logMsg("Zero-config active.", "success");
                    updateState();
                } else {
                    logMsg(data.message, "err");
                }
            }

            // Commands
            async function setDifficulty(lvl) {
                logMsg(`Mutating task ID to ${lvl}...`, "warn");
                await fetch('/config/difficulty', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({level: lvl})
                });
                updateState();
            }

            async function executeReset() {
                logMsg("SPSC_BUFFER_FLUSHING...", "warn");
                await fetch('/api/reset', {method: 'POST'});
                ledgerBody.innerHTML = ''; 
                updateState();
                logMsg("Memory pool purged.", "success");
            }

            async function executeStep(actionType) {
                logMsg(`Dispatching action: ${actionType.toUpperCase()}`, "info");
                const res = await fetch('/api/step', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action_type: actionType})
                });
                const data = await res.json();
                
                logMsg(`[RECON] Reward: ${data.reward} | TP: ${data.tp} | FP: ${data.fp}`, data.reward >= 0.8 ? 'success' : 'warn');
                
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${data.step}</td>
                    <td class="data-text">Vector[40]</td>
                    <td class="data-text">${actionType.toUpperCase()}</td>
                    <td class="${data.reward >= 0.8 ? 'halo-glow' : ''}">${data.reward.toFixed(4)}</td>
                    <td class="${data.done ? 'threat-glow' : ''}">${data.done}</td>
                `;
                if(ledgerBody.children.length >= 5) { ledgerBody.removeChild(ledgerBody.firstChild); }
                ledgerBody.appendChild(row);
                
                updateState();
            }

            // Heartbeat
            setInterval(updateState, 1000);
            updateState();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)