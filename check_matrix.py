import numpy as np
import hft_auditor

engine = hft_auditor.ReconciliationEngine(100)
print("Engine created.")
matrix = engine.get_anomaly_matrix()
print(f"Matrix shape: {matrix.shape}")
print(f"Matrix type: {type(matrix)}")
