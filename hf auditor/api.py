from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import hft_auditor
import os
import sys
import numpy as np

# Ensure the .so/.pyd is discoverable if it was built in /app/build
# In a Docker container, we usually place it in the same directory or a known path.
sys.path.append(os.getcwd())

app = FastAPI(title="HFT Risk Compliance Engine API")

# Initialize Engine with 1M slots
# Note: In a production environment, this would be managed as a singleton.
engine = hft_auditor.ReconciliationEngine(1048576)

class Trade(BaseModel):
    trade_id: int
    price: int
    quantity: int
    counterparty_id: int
    timestamp_ns: int

class BankReceipt(BaseModel):
    trade_id: int
    expected_price: int
    expected_qty: int

@app.post("/ingest")
async def ingest_trade(trade: Trade):
    """Submit a trade to the lock-free SPSC ring buffer."""
    success = engine.submit_trade(
        trade.trade_id, 
        trade.price, 
        trade.quantity, 
        trade.counterparty_id, 
        trade.timestamp_ns
    )
    if not success:
        raise HTTPException(status_code=503, detail="Ingestion buffer full (back-pressure)")
    return {"status": "enqueued", "trade_id": trade.trade_id}

@app.post("/tick")
async def tick_engine(watermark_ns: int):
    """Advance the engine watermark and process enqueued trades."""
    ingested = engine.tick(watermark_ns)
    return {
        "status": "ticked", 
        "ingested_count": ingested, 
        "current_watermark": engine.watermark
    }

@app.post("/reconcile")
async def reconcile_trade(receipt: BankReceipt):
    """Match a bank receipt against the internal trade pool (O(1))."""
    result = engine.reconcile(
        receipt.trade_id, 
        receipt.expected_price, 
        receipt.expected_qty
    )
    # ReconcileResult mapping: 0=MATCH, 1=PRICE_MISMATCH, 2=QTY_MISMATCH, 3=NOT_FOUND, 4=ALREADY_DONE
    results_map = {
        0: "MATCH",
        1: "PRICE_MISMATCH",
        2: "QTY_MISMATCH",
        3: "NOT_FOUND",
        4: "ALREADY_DONE"
    }
    return {"status": results_map.get(result, "UNKNOWN"), "code": result}

@app.get("/stats")
async def get_stats():
    """Get real-time engine statistics."""
    return {
        "total_ingested": engine.total_ingested,
        "total_reconciled": engine.total_reconciled,
        "total_expired": engine.total_expired,
        "active_trades": engine.active_count,
        "ring_buffer_backlog": engine.ring_buffer_size,
        "watermark_ns": engine.watermark
    }

@app.get("/observations")
async def get_observations():
    """Get the zero-copy observation matrix for RL agents."""
    obs = engine.get_observation_matrix()
    return {"shape": obs.shape, "data": obs.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
