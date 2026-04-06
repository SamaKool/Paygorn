import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from typing import Optional

# We import the environment first. 
# Python will find 'hft_auditor' automatically via the Docker PYTHONPATH.
from server.fin_auditor_environment import FinAuditorEnvironment
from models import AuditorAction, AuditorObservation

app = FastAPI(
    title="PayGorn HFT Auditor",
    description="Engineered for high-frequency compliance. Powered by C++20.",
    version="1.0.0"
)

# Global Instance: Shared between API and Dashboard
env = FinAuditorEnvironment()

@app.post("/reset")
async def reset():
    return env.reset()

@app.post("/step")
async def step(action: AuditorAction):
    return env.step(action)

@app.get("/state")
async def get_state():
    return {
        "step_count": env.state.step_count,
        "difficulty": str(env.difficulty),
        "metrics": {
            "tp": getattr(env.state, 'last_tp', 0),
            "tn": getattr(env.state, 'last_tn import ', 0)
        }
    }

@app.post("/config")
async def update_config(task_id: str):
    """Wires the Dashboard buttons to the C++ Engine."""
    import hft_auditor # Standard import - safe here
    os.environ["TASK_ID"] = task_id
    
    if "easy" in task_id.lower():
        env.difficulty = hft_auditor.Difficulty.EASY
    elif "medium" in task_id.lower():
        env.difficulty = hft_auditor.Difficulty.MEDIUM
    else:
        env.difficulty = hft_auditor.Difficulty.HARD
        
    return {"status": "success", "active_difficulty": task_id}

@app.get("/", response_class=HTMLResponse)
async def root_dashboard():
    # ... [Keep your existing CSS here] ...
    html_content = """
    <script>
        async function setDifficulty(id) {
            // Updated to use the Query Parameter for /config
            const res = await fetch(`/config?task_id=${id}`, {method: 'POST'});
            const data = await res.json();
            document.getElementById('diff-label').innerText = id.split('_').pop().toUpperCase();
            console.log(`Difficulty updated: ${data.active_difficulty}`);
        }

        async function resetEnv() {
            await fetch('/reset', {method: 'POST'});
            location.reload(); // Hard reset for the UI
        }
        
        // FIX: Ensure API Docs opens correctly relative to the Space URL
        function openDocs() {
            window.open('./docs', '_blank');
        }
    </script>
    <button class="btn" onclick="setDifficulty('anomaly_detection_easy')">Easy Mode</button>
    <button class="btn" onclick="setDifficulty('anomaly_detection_medium')">Medium Risk</button>
    <button class="btn" onclick="setDifficulty('anomaly_detection_hard')">Hard Adversarial</button>
    
    <button class="btn" onclick="openDocs()">Developer API Reference</button>
    """
    return HTMLResponse(content=html_content)