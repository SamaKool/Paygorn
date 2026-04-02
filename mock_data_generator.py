import pandas as pd
import numpy as np
import json
from uuid import uuid4

def generate_data(n=10000):
    # 1. Internal Trades (The Source of Truth)
    trades = []
    for i in range(n):
        trades.append({
            "trade_id": str(uuid4())[:8],
            "amount": round(np.random.uniform(1000, 50000), 2),
            "timestamp": 1712000000 + i, # Event Time
            "counterparty_id": np.random.randint(100, 500)
        })
    
    # 2. Bank Receipts (The External Stream - with errors)
    receipts = []
    for t in trades:
        # Create some intentional discrepancies (Anomalies)
        chance = np.random.random()
        amount = t["amount"]
        timestamp = t["timestamp"] + np.random.randint(0, 10) # Asynchronous delay

        if chance < 0.05: # 5% Price Mismatch
            amount += np.random.uniform(-10, 10)
        elif chance < 0.10: # 5% Major Delay
            timestamp += 5000 

        receipts.append({"trade_id": t["trade_id"], "amount": amount, "timestamp": timestamp})

    pd.DataFrame(trades).to_csv("internal_trades.csv", index=False)
    with open("bank_receipts.json", "w") as f:
        json.dump(receipts, f)

if __name__ == "__main__":
    generate_data()
    print("Mock CSV and JSON generated.")