import os
import sys
import numpy as np

# Add the build directory to path to find the .pyd
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
sys.path.append(build_dir)

# MinGW-compiled .pyd needs GCC runtime DLLs (libstdc++-6.dll, libgcc_s_seh-1.dll)
mingw_bin = r"D:\mingw64\bin"
if os.path.isdir(mingw_bin):
    os.add_dll_directory(mingw_bin)

try:
    import hft_auditor
    print("Successfully imported hft_auditor")
except ImportError as e:
    print(f"FAILED to import hft_auditor: {e}")
    sys.exit(1)

def test_reward_system():
    print("\n" + "="*60)
    print("VERIFYING ASYMMETRIC REWARD & DIFFICULTY MODES")
    print("="*60)
    
    engine = hft_auditor.ReconciliationEngine(1024)
    
    # ── TEST 1: EASY MODE ──
    print("\n[TEST 1] Easy Mode Verification")
    engine.set_seed(42)
    # Generate 100 trades. In EASY mode, counterparty >= 50 is an anomaly.
    target_anomalies = engine.generate_batch(hft_auditor.Difficulty.EASY, 100, 0)
    print(f"  Generated 100 trades. Ground-truth anomalies: {target_anomalies}")
    
    # Expire them all
    engine.tick(6_000_000_000) # Past 5s delta_max
    print(f"  Expired count: {engine.last_expired_count}")
    assert engine.last_expired_count == 100
    
    # Get anomaly matrix to check risk_scores
    mat = engine.get_anomaly_matrix()
    
    # Perfect Agent: flags if risk_score >= 0.5
    actions = np.array([1 if row[3] >= 0.50 else 0 for row in mat], dtype=np.uint8)
    
    reward = engine.compute_reward(actions)
    print(f"  Perfect Agent Reward: {reward:.2f}")
    print(f"  Stats: TP={engine.last_tp}, TN={engine.last_tn}, FP={engine.last_fp}, FN={engine.last_fn}")
    
    # Check asymmetry (TP=1.0, TN=0.5, FP=0.1, FN=0.0)
    expected_reward = (engine.last_tp * 1.0) + (engine.last_tn * 0.5)
    assert abs(reward - expected_reward) < 1e-4
    assert engine.last_fn == 0, "Perfect agent should have 0 FN in EASY mode"
    assert engine.last_fp == 0, "Perfect agent should have 0 FP in EASY mode"
    print("  [PASS] Easy mode perfect agent")

    # ── TEST 2: ADVERSARIAL REWARD ──
    print("\n[TEST 2] Reward Penalty Verification")
    # Intentional False Negative (Missed anomaly)
    # actions_miss: flag nothing
    actions_miss = np.zeros(100, dtype=np.uint8)
    # We need to regenerate because get_anomaly_matrix() cleared the pool
    engine.set_seed(42)
    engine.generate_batch(hft_auditor.Difficulty.EASY, 100, 10_000_000_000)
    engine.tick(20_000_000_000)
    
    reward_miss = engine.compute_reward(actions_miss)
    print(f"  Lazy Agent Reward (Missed everything): {reward_miss:.2f}")
    print(f"  Stats: TP={engine.last_tp}, TN={engine.last_tn}, FP={engine.last_fp}, FN={engine.last_fn}")
    # TNs yield 0.5, FNs yield 0.0
    expected_miss = (engine.last_tn * 0.5) + (engine.last_fn * 0.0)
    assert abs(reward_miss - expected_miss) < 1e-4
    assert engine.last_fn == target_anomalies
    print("  [PASS] False negative penalty logic")

    # ── TEST 3: HARD MODE ──
    print("\n[TEST 3] Hard Mode Distribution")
    engine.set_seed(123)
    # Hard mode should have non-deterministic labels relative to risk score
    n_hard = engine.generate_batch(hft_auditor.Difficulty.HARD, 1000, 30_000_000_000)
    print(f"  Hard Mode Anomalies (1000 trades): {n_hard}")
    # Rate should be roughly 15-25% (20% expected for HARD mode)
    assert 150 <= n_hard <= 250
    print("  [PASS] Hard mode distribution")

    print("\n" + "="*60)
    print("VERIFICATION SUCCESSFUL")
    print("="*60)

if __name__ == "__main__":
    test_reward_system()
