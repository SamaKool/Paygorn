import os
import sys
import numpy as np

# Add DLL directory for MinGW on Windows if needed
if os.name == 'nt':
    # Add common MinGW paths just in case
    for path in [r"D:\mingw64\bin", r"C:\msys64\mingw64\bin"]:
        if os.path.exists(path):
            os.add_dll_directory(path)

# Add the build directory to path to find the .pyd
sys.path.append(os.path.join(os.getcwd(), "build"))

try:
    import hft_auditor
    print("Successfully imported hft_auditor")
except ImportError as e:
    print(f"FAILED to import hft_auditor: {e}")
    sys.exit(1)

def test_anomaly_matrix():
    print("\n" + "="*60)
    print("VERIFYING ANOMALY MATRIX (Phas 4 Bridge)")
    print("="*60)
    
    # Initialize engine with 1024 slots
    engine = hft_auditor.ReconciliationEngine(1024)
    
    # 1. Ingest 10 trades at timestamp 0
    # trade_id, price, qty, counterparty, ts_ns
    for i in range(10):
        engine.ingest_trade(i, 1000 + i, 50, 77, 0)
    
    print(f"Ingested 10 trades. Active count: {engine.active_count}")
    
    # 2. Tick below expiry (e.g. 4 seconds)
    engine.tick(4_000_000_000)
    print(f"Tick to 4s. Last expired count: {engine.last_expired_count}")
    
    # Anomaly matrix should be empty
    ano_mat = engine.get_anomaly_matrix()
    print(f"Anomaly matrix shape: {ano_mat.shape}")
    assert ano_mat.shape == (0, 4)
    
    # 3. Tick past expiry (e.g. 6 seconds)
    # The Δ_max is 5.0 seconds
    engine.tick(6_000_000_000)
    print(f"Tick to 6s. Last expired count: {engine.last_expired_count}")
    assert engine.last_expired_count == 10
    
    # 4. Inspect anomaly matrix
    ano_mat = engine.get_anomaly_matrix()
    print(f"Anomaly matrix shape: {ano_mat.shape}")
    assert ano_mat.shape == (10, 4)
    
    # Check features for the first row
    # Col 0: time_elapsed = (watermark - ts) / 5e9 = (6e9 - 0) / 5e9 = 1.2
    # Col 1: price_delta = 0.0
    # Col 2: missing_freq = total_expired / total_ingested = 10 / 10 = 1.0
    # Col 3: risk_score = (77 % 100) / 100.0 = 0.77
    
    first_row = ano_mat[0]
    print(f"First row features: {first_row}")
    
    np.testing.assert_almost_equal(first_row[0], 1.2, decimal=3)
    assert first_row[1] == 0.0
    assert first_row[2] == 1.0
    assert first_row[3] == 0.77
    
    print("\n[PASS] Anomaly matrix verification successful.")

if __name__ == "__main__":
    test_anomaly_matrix()
