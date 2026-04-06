import os
import uvicorn
import glob
import importlib.util
import sys
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional

# --- STEP 1: DYNAMIC C++ ENGINE LOADING ---
# Ensures the hft_auditor module is available globally
so_files = glob.glob("**/hft_auditor*.so", recursive=True)
if so_files:
    engine_path = os.path.abspath(so_files[0])
    spec = importlib.util.spec_from_file_location("hft_auditor", engine_path)
    hft_auditor = importlib.util.module_from_spec(spec)
    sys.modules["hft_auditor"] = hft_auditor
    spec.loader.exec_module(hft_auditor)
    print(f"✅ Engine loaded from: {engine_path}")

from server.fin_auditor_environment import FinAuditorEnvironment
from models import AuditorAction, AuditorObservation

app = FastAPI(
    title="PayGorn HFT Auditor",
    description="Engineered for high-frequency compliance. Powered by C++20.",
    version="1.0.0"
)

# Global Environment Instance
env = FinAuditorEnvironment()

# --- API ENDPOINTS FOR AGENT & DASHBOARD ---

@app.post("/reset")
async def reset():
    obs = env.reset()
    return obs

@app.post("/step")
async def step(action: AuditorAction):
    obs = env.step(action)
    return obs

@app.get("/state")
async def get_state():
    # Expose raw engine metrics to the dashboard
    return {
        "episode_id": env.state.episode_id,
        "step_count": env.state.step_count,
        "difficulty": str(env.difficulty),
        "metrics": {
            "tp": getattr(env.state, 'last_tp', 0),
            "tn": getattr(env.state, 'last_tn', 0),
            "fp": getattr(env.state, 'last_fp', 0),
            "fn": getattr(env.state, 'last_fn', 0)
        }
    }

@app.post("/config")
async def update_config(task_id: str):
    """Dynamically updates the TASK_ID difficulty for the next batch."""
    os.environ["TASK_ID"] = task_id
    # Re-initialize engine difficulty logic
    if "easy" in task_id.lower():
        env.difficulty = hft_auditor.Difficulty.EASY
    elif "medium" in task_id.lower():
        env.difficulty = hft_auditor.Difficulty.MEDIUM
    else:
        env.difficulty = hft_auditor.Difficulty.HARD
    return {"status": "success", "active_difficulty": task_id}

# --- THE PSYCHOLOGICALLY OPTIMIZED DASHBOARD ---

@app.get("/", response_class=HTMLResponse)
async def root_dashboard():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>PayGorn | HFT Command Center</title>
        <style>
            :root {
                --bg: #0B0E14; --sidebar: #0D1117; --surface: #151921;
                --border: #30363d; --text-main: #E6EDF3; --text-dim: #8B949E;
                --accent-blue: #2F81F7; --accent-emerald: #00FF9D; --accent-red: #F85149;
            }
            body { background: var(--bg); color: var(--text-main); font-family: sans-serif; margin: 0; display: flex; height: 100vh; overflow: hidden; }
            .sidebar { width: 280px; background: var(--sidebar); border-right: 1px solid var(--border); padding: 32px 24px; }
            .main { flex: 1; overflow-y: auto; padding: 40px 60px; }
            .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; margin-bottom: 32px; }
            .card { background: var(--surface); border: 1px solid var(--border); padding: 24px; border-radius: 12px; }
            .val { font-size: 32px; font-weight: 700; color: var(--accent-emerald); }
            .status-dot { width: 8px; height: 8px; background: var(--accent-emerald); border-radius: 50%; margin-right: 10px; box-shadow: 0 0 10px var(--accent-emerald); animation: pulse 2s infinite; }
            @keyframes pulse { 0%, 100% { opacity: 0.4; } 50% { opacity: 1; } }
            .btn { background: #21262d; border: 1px solid var(--border); color: var(--text-main); padding: 12px 24px; border-radius: 8px; cursor: pointer; }
            .btn-primary { background: var(--accent-blue); border: none; }
            .console { background: #000; border-radius: 8px; padding: 20px; font-family: monospace; font-size: 13px; color: #888; margin-top: 32px; height: 150px; overflow-y: auto; }
        </style>
    </head>
    <body>
        <aside class="sidebar">
            <h2 style="color:var(--accent-blue)">PAYGORN.AI</h2>
            <p style="font-size:10px; color:var(--text-dim)">HFT COMPLIANCE V1.0</p>
        </aside>
        <main class="main">
            <header style="display:flex; justify-content:space-between">
                <div><h2>Operations Center</h2></div>
                <div style="display:flex; align-items:center"><div class="status-dot"></div>SYSTEM: NATIVE_RECON_ONLINE</div>
            </header>
            <section class="grid">
                <div class="card"><h3>Step Count</h3><div id="step-count" class="val">0</div></div>
                <div class="card"><h3>Active Difficulty</h3><div id="diff-label" class="val" style="color:var(--accent-blue)">HARD</div></div>
                <div class="card"><h3>Last Batch Reward</h3><div id="reward-val" class="val">0.00</div></div>
            </section>
            <section class="card">
                <h3>Set Task Difficulty</h3>
                <div style="display:flex; gap:10px">
                    <button class="btn" onclick="setDifficulty('anomaly_detection_easy')">Easy</button>
                    <button class="btn" onclick="setDifficulty('anomaly_detection_medium')">Medium</button>
                    <button class="btn" onclick="setDifficulty('anomaly_detection_hard')">Hard</button>
                </div>
            </section>
            <div id="console" class="console">[SYSTEM] HFT Engine Warm-up Complete...</div>
            <footer style="margin-top:40px">
                <button class="btn btn-primary" onclick="resetEnv()">Execute Full System Reset</button>
                <button class="btn" onclick="window.open('./docs', '_blank')">API Documentation</button>
            </footer>
        </main>
        <script>
            async function setDifficulty(id) {
                const res = await fetch(`/config?task_id=${id}`, {method: 'POST'});
                const data = await res.json();
                document.getElementById('diff-label').innerText = id.split('_').pop().toUpperCase();
                logConsole(`Difficulty shifted to ${id}`);
            }
            async function resetEnv() {
                await fetch('/reset', {method: 'POST'});
                logConsole('System Reset Triggered. Episode ID re-initialized.');
                updateStats();
            }
            async function updateStats() {
                const res = await fetch('/state');
                const data = await res.json();
                document.getElementById('step-count').innerText = data.step_count;
                document.getElementById('reward-val').innerText = (data.metrics.tp / 40).toFixed(2);
            }
            function logConsole(msg) {
                const c = document.getElementById('console');
                c.innerHTML += `<br>[${new Date().toLocaleTimeString()}] ${msg}`;
                c.scrollTop = c.scrollHeight;
            }
            setInterval(updateStats, 2000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)