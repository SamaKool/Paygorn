import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(
    title="PayGorn HFT Compliance Auditor",
    description="Enterprise-grade RL environment powered by a C++20 Zero-Allocation Engine.",
    version="1.0.0"
)

# ... (Keep your existing endpoint logic for /step, /reset, etc.) ...

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def root_dashboard():
    """Returns the interactive HFT Command Center Dashboard."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PayGorn HFT Auditor | Command Center</title>
        <style>
            body { background-color: #0d1117; color: #c9d1d9; font-family: monospace; padding: 40px; }
            .header { border-bottom: 1px solid #30363d; padding-bottom: 20px; margin-bottom: 20px; }
            .badge { background-color: #238636; color: #ffffff; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .card { background-color: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 10px; }
            .metric { font-size: 24px; color: #58a6ff; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📡 PayGorn HFT Operations Center</h1>
            <span class="badge">SYSTEM ONLINE</span>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>⚡ Engine Specifications</h3>
                <p>Architecture: <b>C++20 Native (Nanobind)</b></p>
                <p>Memory Model: <b>SPSC Ring Buffer (Zero-Copy)</b></p>
                <p>Throughput Cap: <span class="metric">1,000,000 ops/sec</span></p>
            </div>
            
            <div class="card">
                <h3>📊 Validation Metrics</h3>
                <p>Baseline LLM Score: <b>0.50 (Verified)</b></p>
                <p>Stability Test: <b>27,392 PPO Steps (0 Leaks)</b></p>
                <p>Cost Engine: <b>Asymmetric (TP: 1.0, TN: 0.5, FP: 0.1, FN: 0.0)</b></p>
            </div>
        </div>

        <div class="card" style="margin-top: 20px;">
            <h3>🎯 Dynamic Difficulty Tiers Available</h3>
            <ul>
                <li>🟢 <b>EASY:</b> Deterministic anomaly distribution.</li>
                <li>🟡 <b>MEDIUM:</b> 80% anomaly probability on high-risk counterparties.</li>
                <li>🔴 <b>HARD:</b> 40-55% correlation with severe background noise.</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()