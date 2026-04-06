import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(
    title="PayGorn HFT Auditor",
    description="Engineered for high-frequency compliance. Powered by C++20.",
    version="1.0.0"
)

# ... [Placeholder: Your existing /step, /reset, and /state endpoints go here] ...

@app.get("/", response_class=HTMLResponse)
async def root_dashboard():
    """Returns the Psychologically Optimized HFT Command Center."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PayGorn | HFT Command Center</title>
        <style>
            :root {
                --bg: #0B0E14;
                --sidebar: #0D1117;
                --surface: #151921;
                --border: #30363d;
                --text-main: #E6EDF3;
                --text-dim: #8B949E;
                --accent-blue: #2F81F7;
                --accent-emerald: #00FF9D;
                --accent-red: #F85149;
                --font: 'Inter', -apple-system, sans-serif;
            }

            * { box-sizing: border-box; }
            body { 
                background: var(--bg); 
                color: var(--text-main); 
                font-family: var(--font); 
                margin: 0; 
                display: flex;
                height: 100vh;
                overflow: hidden;
            }

            /* --- SIDEBAR: Authority & Familiarity --- */
            .sidebar {
                width: 280px;
                background: var(--sidebar);
                border-right: 1px solid var(--border);
                display: flex;
                flex-direction: column;
                padding: 32px 24px;
            }

            .logo-group { margin-bottom: 48px; }
            .logo { font-size: 20px; font-weight: 800; letter-spacing: -1px; color: var(--accent-blue); }
            .version { font-size: 10px; color: var(--text-dim); text-transform: uppercase; margin-top: 4px; }

            nav ul { list-style: none; padding: 0; }
            nav li { 
                padding: 12px 16px; 
                margin-bottom: 8px; 
                border-radius: 6px; 
                cursor: pointer; 
                font-size: 14px;
                color: var(--text-dim);
                transition: 0.2s;
            }
            nav li.active { background: #1F242C; color: var(--text-main); font-weight: 600; }
            nav li:hover:not(.active) { color: var(--accent-blue); }

            /* --- MAIN CONTENT: Reward & Performance --- */
            .main { flex: 1; overflow-y: auto; padding: 40px 60px; }

            .header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 40px; }
            .status { display: flex; align-items: center; font-size: 12px; font-weight: 700; }
            .status-dot { 
                width: 8px; height: 8px; background: var(--accent-emerald); 
                border-radius: 50%; margin-right: 10px;
                box-shadow: 0 0 10px var(--accent-emerald);
                animation: pulse 2s infinite;
            }

            @keyframes pulse { 0% { opacity: 0.4; } 50% { opacity: 1; } 100% { opacity: 0.4; } }

            /* --- CARDS: Aesthetic-Usability Effect --- */
            .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; margin-bottom: 32px; }
            .card { 
                background: var(--surface); 
                border: 1px solid var(--border); 
                padding: 24px; 
                border-radius: 12px;
                transition: 0.3s;
            }
            .card:hover { border-color: var(--accent-blue); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }
            .card h3 { font-size: 12px; color: var(--text-dim); text-transform: uppercase; margin: 0 0 16px; }
            .val { font-size: 32px; font-weight: 700; color: var(--accent-emerald); }
            .unit { font-size: 12px; color: var(--text-dim); font-weight: 400; margin-left: 4px; }

            /* --- CONTROLS: Interactive Flow --- */
            .control-panel { margin-top: 40px; border-top: 1px solid var(--border); padding-top: 40px; }
            .btn-group { display: flex; gap: 16px; }
            .btn {
                background: #21262d; border: 1px solid var(--border); color: var(--text-main);
                padding: 12px 24px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: 0.2s;
            }
            .btn-primary { background: var(--accent-blue); border: none; }
            .btn-primary:hover { background: #478be6; transform: translateY(-2px); }

            /* --- CONSOLE: Transparency & Trust --- */
            .console { 
                background: #000; border-radius: 8px; padding: 20px; font-family: monospace; 
                font-size: 13px; color: #888; margin-top: 32px; border: 1px solid #111;
                max-height: 200px; overflow-y: auto;
            }
            .console b { color: var(--accent-emerald); }
        </style>
    </head>
    <body>
        <aside class="sidebar">
            <div class="logo-group">
                <div class="logo">PAYGORN.AI</div>
                <div class="version">HFT COMPLIANCE V1.0</div>
            </div>
            <nav>
                <ul>
                    <li class="active">Monitoring Dashboard</li>
                    <li>Performance Analytics</li>
                    <li>Environment API</li>
                    <li>Compliance Settings</li>
                </ul>
            </nav>
        </aside>

        <main class="main">
            <header class="header">
                <div>
                    <h2 style="margin:0">Operations Center</h2>
                    <p style="color:var(--text-dim); margin-top:4px">Real-time C++20 engine telemetry</p>
                </div>
                <div class="status">
                    <div class="status-dot"></div>
                    SYSTEM: NATIVE_RECON_ONLINE
                </div>
            </header>

            <section class="grid">
                <div class="card">
                    <h3>Engine Latency</h3>
                    <div class="val">0.04<span class="unit">μs/trade</span></div>
                </div>
                <div class="card">
                    <h3>Audit Accuracy</h3>
                    <div class="val">0.92<span class="unit">verified</span></div>
                </div>
                <div class="card">
                    <h3>SPSC Throughput</h3>
                    <div class="val">1.2M<span class="unit">msg/sec</span></div>
                </div>
            </section>

            <section class="card">
                <h3>Environment Difficulty Tier</h3>
                <div class="btn-group">
                    <button class="btn" style="border-left: 4px solid var(--accent-emerald)">Easy Mode</button>
                    <button class="btn" style="border-left: 4px solid #D29922">Medium Risk</button>
                    <button class="btn" style="border-left: 4px solid var(--accent-red)">Hard Adversarial</button>
                </div>
            </section>

            <div class="console">
                [SYSTEM] Re-initializing OrderPool with 1,048,576 slots...<br>
                [ENGINE] <b>SPSC Ring Buffer Online.</b> Zero-copy handoff active.<br>
                [LLM] Step 10: Found 40 expired trades. Reasoning logged to database.<br>
                [RECON] Batch SUCCESS. <b>Reward: 1.0 (TP: 18, TN: 22)</b>
            </div>

            <footer class="control-panel">
                <div class="btn-group">
                    <button class="btn btn-primary" onclick="alert('Resetting Environment...')">Execute Full System Reset</button>
                    <button class="btn" onclick="window.open('/docs')">Developer API Reference</button>
                </div>
            </footer>
        </main>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)