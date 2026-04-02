"""
Integration test for the ReconciliationEngine C++ backend.

Tests:
  1. Engine initialization and capacity
  2. Direct ingestion (single-threaded mode)
  3. Timer wheel expiration (Δ_max = 5.0 seconds)
  4. Reconciliation (match, price mismatch, not found)
  5. Zero-copy observation matrix (shape, dtype, values)
  6. SPSC ring buffer + tick() pipeline
  7. Statistics properties
"""

import sys
import os
import numpy as np

# Add the build directory to Python's module search path
build_dir = os.path.join(os.path.dirname(__file__), "build")
sys.path.insert(0, build_dir)

# MinGW-compiled .pyd needs GCC runtime DLLs (libstdc++-6.dll, libgcc_s_seh-1.dll)
# os.add_dll_directory() makes them discoverable without polluting system PATH
mingw_bin = r"D:\mingw64\bin"
if os.path.isdir(mingw_bin):
    os.add_dll_directory(mingw_bin)

import hft_auditor

# ── Constants ────────────────────────────────────────────────────────────
NS_PER_SEC = 1_000_000_000
DELTA_MAX_NS = 5 * NS_PER_SEC  # 5 seconds


def test_initialization():
    """Test engine creation and capacity rounding."""
    print("=" * 60)
    print("TEST 1: Initialization")
    print("=" * 60)

    engine = hft_auditor.ReconciliationEngine(capacity=1_000_000)
    print(f"  Pool capacity: {engine.pool_capacity}")
    print(f"  Active count:  {engine.active_count}")
    print(f"  Watermark:     {engine.watermark}")

    # Capacity should be rounded up to next power of 2
    assert engine.pool_capacity == 1_048_576, \
        f"Expected 1048576, got {engine.pool_capacity}"
    assert engine.active_count == 0
    assert engine.watermark == 0
    print("  [PASS]\n")
    return engine


def test_direct_ingestion(engine):
    """Test direct trade ingestion (bypassing ring buffer)."""
    print("=" * 60)
    print("TEST 2: Direct Ingestion")
    print("=" * 60)

    base_time = 1_000_000_000_000  # 1000 seconds in ns

    # Ingest 1000 trades
    for i in range(1000):
        ok = engine.ingest_trade(
            trade_id=i,
            price=10000 + i,
            quantity=100 + (i % 10),
            counterparty_id=i % 50,
            timestamp_ns=base_time + i * 1000  # 1μs apart
        )
        assert ok, f"ingest_trade failed for trade {i}"

    print(f"  Total ingested:  {engine.total_ingested}")
    print(f"  Active count:    {engine.active_count}")
    assert engine.total_ingested == 1000
    assert engine.active_count == 1000
    print("  [PASS]\n")
    return base_time


def test_observation_matrix(engine):
    """Test zero-copy observation matrix."""
    print("=" * 60)
    print("TEST 3: Observation Matrix (Zero-Copy)")
    print("=" * 60)

    obs = engine.get_observation_matrix()
    obs_np = np.asarray(obs)

    print(f"  Shape:  {obs_np.shape}")
    print(f"  Dtype:  {obs_np.dtype}")
    print(f"  First row: {obs_np[0] if len(obs_np) > 0 else 'empty'}")
    print(f"  Last row:  {obs_np[-1] if len(obs_np) > 0 else 'empty'}")

    assert obs_np.shape == (1000, 4), f"Expected (1000, 4), got {obs_np.shape}"
    assert obs_np.dtype == np.float32, f"Expected float32, got {obs_np.dtype}"

    # Check risk_score column: (counterparty_id % 100) / 100.0
    # Trade 0 has counterparty_id=0 → risk = 0.0
    # Trade 1 has counterparty_id=1 → risk = 0.01
    # Trade 49 has counterparty_id=49 → risk = 0.49
    assert abs(obs_np[0, 3] - 0.00) < 1e-6, f"risk[0] = {obs_np[0, 3]}"
    assert abs(obs_np[1, 3] - 0.01) < 1e-6, f"risk[1] = {obs_np[1, 3]}"
    assert abs(obs_np[49, 3] - 0.49) < 1e-6, f"risk[49] = {obs_np[49, 3]}"
    print("  [PASS]\n")


def test_reconciliation(engine):
    """Test reconciliation: match, mismatch, not-found."""
    print("=" * 60)
    print("TEST 4: Reconciliation")
    print("=" * 60)

    # Perfect match: trade 0 has price=10000, qty=100
    result = engine.reconcile(trade_id=0, expected_price=10000, expected_qty=100)
    print(f"  Trade 0 (exact match):     result={result}")
    assert result == 0, f"Expected MATCH(0), got {result}"

    # Price mismatch: trade 1 has price=10001, but we pass 99999
    result = engine.reconcile(trade_id=1, expected_price=99999, expected_qty=101)
    print(f"  Trade 1 (price mismatch):  result={result}")
    assert result == 1, f"Expected PRICE_MISMATCH(1), got {result}"

    # Qty mismatch: trade 2 has price=10002, qty=102, but we pass qty=999
    result = engine.reconcile(trade_id=2, expected_price=10002, expected_qty=999)
    print(f"  Trade 2 (qty mismatch):    result={result}")
    assert result == 2, f"Expected QTY_MISMATCH(2), got {result}"

    # Try to reconcile trade 0 again (already reconciled)
    result = engine.reconcile(trade_id=0, expected_price=10000, expected_qty=100)
    print(f"  Trade 0 (already done):    result={result}")
    assert result == 4, f"Expected ALREADY_DONE(4), got {result}"

    # Not found: trade ID that doesn't exist
    result = engine.reconcile(trade_id=9999999, expected_price=0, expected_qty=0)
    print(f"  Trade 9999999 (not found): result={result}")
    assert result == 3, f"Expected NOT_FOUND(3), got {result}"

    print(f"  Total reconciled: {engine.total_reconciled}")
    assert engine.total_reconciled == 1  # Only trade 0 was a perfect match
    print("  [PASS]\n")


def test_timer_wheel_expiration():
    """Test that trades expire after Delta_max watermark advance."""
    print("=" * 60)
    print("TEST 5: Timer Wheel Expiration (Delta_max = 5.0s)")
    print("=" * 60)

    engine = hft_auditor.ReconciliationEngine(capacity=1024)

    base_time = 1_000_000_000_000  # 1000s in ns

    # Ingest 100 trades at base_time
    for i in range(100):
        engine.ingest_trade(
            trade_id=i,
            price=5000 + i,
            quantity=10,
            counterparty_id=i % 20,
            timestamp_ns=base_time
        )

    print(f"  After ingest:    active={engine.active_count}, expired={engine.total_expired}")
    assert engine.active_count == 100

    # Advance watermark by 2 seconds — no expirations
    engine.tick(base_time + 2 * NS_PER_SEC)
    print(f"  After +2.0s:     active={engine.active_count}, expired={engine.total_expired}")
    assert engine.active_count == 100, f"Expected 100, got {engine.active_count}"
    assert engine.total_expired == 0

    # Advance watermark by 4.9 seconds — still no expirations
    engine.tick(base_time + int(4.9 * NS_PER_SEC))
    print(f"  After +4.9s:     active={engine.active_count}, expired={engine.total_expired}")
    assert engine.active_count == 100, f"Expected 100, got {engine.active_count}"
    assert engine.total_expired == 0

    # Advance watermark past Δ_max (5.0 seconds) — ALL should expire
    engine.tick(base_time + int(5.1 * NS_PER_SEC))
    print(f"  After +5.1s:     active={engine.active_count}, expired={engine.total_expired}")
    assert engine.total_expired == 100, f"Expected 100 expired, got {engine.total_expired}"
    assert engine.active_count == 0, f"Expected 0 active, got {engine.active_count}"

    print("  [PASS]\n")


def test_spsc_ring_buffer():
    """Test the SPSC ring buffer + tick() pipeline."""
    print("=" * 60)
    print("TEST 6: SPSC Ring Buffer + Tick()")
    print("=" * 60)

    engine = hft_auditor.ReconciliationEngine(capacity=65536)

    base_time = 2_000_000_000_000  # 2000s in ns

    # Submit trades via the ring buffer
    for i in range(500):
        ok = engine.submit_trade(
            trade_id=i,
            price=8000 + i,
            quantity=50,
            counterparty_id=i % 30,
            timestamp_ns=base_time + i * 10_000  # 10us apart
        )
        assert ok, f"submit_trade failed for trade {i}"

    print(f"  Ring buffer size after submit: {engine.ring_buffer_size}")
    assert engine.ring_buffer_size == 500

    # Tick to drain the ring buffer
    ingested = engine.tick(base_time + 500 * 10_000)
    print(f"  Ingested by tick():           {ingested}")
    print(f"  Ring buffer size after tick:   {engine.ring_buffer_size}")
    print(f"  Active count after tick:       {engine.active_count}")
    assert ingested == 500
    assert engine.ring_buffer_size == 0
    assert engine.active_count == 500

    # Reconcile one trade from the ring buffer batch
    result = engine.reconcile(trade_id=42, expected_price=8042, expected_qty=50)
    print(f"  Reconcile trade 42: result={result}")
    assert result == 0  # MATCH

    print("  [PASS]\n")


def test_performance():
    """Basic throughput test: ingest 1M trades."""
    print("=" * 60)
    print("TEST 7: Performance (1M Ingestion)")
    print("=" * 60)

    import time

    engine = hft_auditor.ReconciliationEngine(capacity=1_048_576)
    base_time = 3_000_000_000_000  # 3000s in ns
    count = 1_000_000

    start = time.perf_counter()
    for i in range(count):
        engine.ingest_trade(
            trade_id=i,
            price=10000,
            quantity=100,
            counterparty_id=i % 100,
            timestamp_ns=base_time + i * 1000
        )
    elapsed = time.perf_counter() - start

    trades_per_sec = count / elapsed
    print(f"  Ingested:        {count:,} trades")
    print(f"  Elapsed:         {elapsed:.3f} seconds")
    print(f"  Throughput:      {trades_per_sec:,.0f} trades/sec")
    print(f"  Active trades:   {engine.active_count:,}")

    # Get observation matrix for all 1M trades
    start = time.perf_counter()
    obs = engine.get_observation_matrix()
    obs_np = np.asarray(obs)
    obs_elapsed = time.perf_counter() - start
    print(f"  Obs matrix:      {obs_np.shape}")
    print(f"  Obs build time:  {obs_elapsed*1000:.1f} ms")

    assert engine.total_ingested == count
    print("  [PASS]\n")


# ── Run all tests ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  HFT Reconciliation Engine -- Integration Tests")
    print("=" * 60 + "\n")

    engine = test_initialization()
    base_time = test_direct_ingestion(engine)
    test_observation_matrix(engine)
    test_reconciliation(engine)
    test_timer_wheel_expiration()
    test_spsc_ring_buffer()
    test_performance()

    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
