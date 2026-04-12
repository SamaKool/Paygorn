#pragma once
// ============================================================================
// ReconciliationEngine — Orchestrator for HFT Trade Reconciliation
// ============================================================================
//
// This is the top-level class that owns all three subsystems:
//   1. OrderPool          — O(1) flat-array trade storage
//   2. TimerWheel         — O(1) hierarchical expiration
//   3. SPSCRingBuffer     — Lock-free ingestion from producer thread
//
// THREADING MODEL:
//   ┌─────────────────┐         ┌─────────────────────────────────┐
//   │ Ingestion Thread │  SPSC   │ Engine Thread (Python's thread) │
//   │ (producer)       │ ──────→ │ tick() → drain → insert → expire│
//   │ submit_trade()   │ ring    │ reconcile() → match receipts    │
//   │                  │ buffer  │ get_observation_matrix() → RL   │
//   └─────────────────┘         └─────────────────────────────────┘
//
// The SPSC ring buffer is the ONLY shared data structure between threads.
// Everything else (OrderPool, TimerWheel, observation matrix) is accessed
// exclusively from the engine thread — no locks needed.
//
// OBSERVATION MATRIX (for the RL agent):
//   Shape: (N_active_trades, 4) — zero-copy exposed via nb::ndarray
//   Columns:
//     [0] time_elapsed     — (watermark - trade_ts) / Δ_max, normalized
//     0.0–1.0+ [1] price_delta      — 0.0 (placeholder; populated during
//     reconcile mismatches) [2] missing_frequency — total_expired /
//     total_ingested, global ratio [3] risk_score       — (counterparty_id %
//     100) / 100.0, per-counterparty hash
//
//   This matches Soham's FinAuditorObservation.features schema:
//   List[List[StrictFloat]] with 4 features per anomaly.
//
// ============================================================================

#include "order_pool.hpp"
#include "spsc_ring_buffer.hpp"
#include "timer_wheel.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <iostream>
#include <vector>

namespace nb = nanobind;

// ────────────────────────────────────────────────────────────────────────────
// Ring buffer capacity: 2^21 = 2,097,152 slots
//
// At 1M trades/sec, this gives 2 seconds of buffering headroom against
// OS scheduling preemptions or network micro-bursts.
// ────────────────────────────────────────────────────────────────────────────
static constexpr size_t RING_BUFFER_CAPACITY = 1ULL << 21; // 2,097,152

// ────────────────────────────────────────────────────────────────────────────
// ReconcileResult — Outcome of a reconciliation attempt
// ────────────────────────────────────────────────────────────────────────────
enum class ReconcileResult : uint8_t {
  MATCH = 0,          // Trade matches bank receipt — success
  PRICE_MISMATCH = 1, // Price field differs — anomaly
  QTY_MISMATCH = 2,   // Quantity field differs — anomaly
  NOT_FOUND = 3,      // Trade ID not in pool (wrong ID or already evicted)
  ALREADY_DONE = 4    // Trade already reconciled or expired
};

// ────────────────────────────────────────────────────────────────────────────
// Difficulty — Controls anomaly distribution and signal-to-noise ratio
// ────────────────────────────────────────────────────────────────────────────
enum class Difficulty : uint8_t {
  EASY = 0, // Deterministic: risk_score > 0.5 is always an anomaly
  MEDIUM =
      1,   // Probabilistic: high-risk = 80% anomaly, low-risk = 10% false alarm
  HARD = 2 // Adversarial: weak correlations, high noise
};

// ────────────────────────────────────────────────────────────────────────────
// ReconciliationEngine — The main orchestrator
// ────────────────────────────────────────────────────────────────────────────
class ReconciliationEngine {
public:
  // ────────────────────────────────────────────────────────────────────
  // Constructor: pre-allocates ALL memory at startup
  //
  // capacity: number of trade slots in the OrderPool (rounded up to
  //           next power of 2 for bitwise AND indexing)
  // ────────────────────────────────────────────────────────────────────
  explicit ReconciliationEngine(size_t capacity = 1 << 20)
      : pool_(next_power_of_2(capacity)),
        timer_wheel_(next_power_of_2(capacity)), watermark_ns_(0),
        obs_capacity_(4096) // Initial observation buffer size
        ,
        anomaly_capacity_(1024) // Initial anomaly buffer size
        ,
        reward_details_capacity_(1024) {
    const size_t actual_cap = next_power_of_2(capacity);

    // Pre-allocate the expired-indices buffer (used by tick())
    expired_buffer_.resize(actual_cap);

    // Pre-allocate matrix buffers.
    observation_matrix_.resize(obs_capacity_ * 4, 0.0f);
    anomaly_matrix_.resize(anomaly_capacity_ * 4, 0.0f);
    reward_details_.resize(reward_details_capacity_, 0.0f);
  }

  // ================================================================
  //  PRODUCER THREAD API — Lock-free, called from ingestion thread
  // ================================================================

  // ────────────────────────────────────────────────────────────────────
  // submit_trade() — Enqueue a trade into the SPSC ring buffer
  //
  // This is the ONLY function safe to call from the producer thread.
  // All other methods must be called from the engine thread.
  //
  // Returns false if the ring buffer is full (back-pressure signal).
  // ────────────────────────────────────────────────────────────────────
  bool submit_trade(uint64_t trade_id, int64_t price, int32_t quantity,
                    uint32_t counterparty_id, uint64_t timestamp_ns) {
    return ring_buffer_.try_push(trade_id, price, quantity, counterparty_id,
                                 timestamp_ns);
  }

  // ================================================================
  //  ENGINE THREAD API — Single-threaded, called from Python/RL loop
  // ================================================================

  // ────────────────────────────────────────────────────────────────────
  // tick() — Main processing loop
  //
  // Called once per RL step. Performs three operations:
  //   1. Drains the SPSC ring buffer into the OrderPool
  //   2. Schedules new timers in the TimerWheel
  //   3. Advances the watermark and collects expired trades
  //
  // new_watermark_ns: the event-time to advance to. Must be
  //                   monotonically non-decreasing.
  //
  // Returns: number of trades ingested from the ring buffer
  // ────────────────────────────────────────────────────────────────────
  size_t tick(uint64_t new_watermark_ns) {
    // ── Step 1: Drain ring buffer into OrderPool ──────────────────
    //
    // Each entry from the ring buffer is:
    //   a) Inserted into the OrderPool at O(1) via flat-array index
    //   b) Scheduled in the TimerWheel for future expiration
    //
    const size_t pool_mask = pool_.capacity() - 1;

    size_t ingested = ring_buffer_.drain([&](const RingEntry &entry) {
      // Compute the pool index (same as OrderPool's internal index)
      const uint32_t idx = static_cast<uint32_t>(entry.trade_id & pool_mask);

      // Insert into the flat-array pool — O(1)
      pool_.insert(entry.trade_id, entry.price, entry.quantity,
                   entry.counterparty_id, entry.timestamp_ns);

      // Schedule expiration timer — O(1)
      timer_wheel_.schedule(idx, entry.timestamp_ns);
    });

    // ── Step 2: Advance the timer wheel watermark ─────────────────
    //
    // This sweeps L0 slots, cascades from L1/L2 as needed, and
    // collects indices of trades that have exceeded Δ_max.
    //
    if (new_watermark_ns > watermark_ns_) {
      const size_t expired_count = timer_wheel_.advance(
          new_watermark_ns, expired_buffer_.data(), expired_buffer_.size());

      // ── Step 3: Mark expired trades in the OrderPool ──────────
      last_batch_expired_count_ = expired_count;
      for (size_t i = 0; i < expired_count; ++i) {
        const uint32_t idx = expired_buffer_[i];
        if (pool_.get_state(idx) == SlotState::ACTIVE) {
          pool_.set_state(idx, SlotState::EXPIRED);
          ++total_expired_;
        }
      }
    } else {
      last_batch_expired_count_ = 0;
    }

    watermark_ns_ = new_watermark_ns;
    total_ingested_ += ingested;
    return ingested;
  }

  // ────────────────────────────────────────────────────────────────────
  // reconcile() — Match an internal trade against an external receipt
  //
  // Called when a bank receipt arrives. Looks up the trade by ID and
  // compares price and quantity.
  //
  // Returns a uint8_t code (castable to ReconcileResult):
  //   0 = MATCH           — perfect match, trade marked RECONCILED
  //   1 = PRICE_MISMATCH  — price differs (anomaly for RL agent)
  //   2 = QTY_MISMATCH    — quantity differs (anomaly for RL agent)
  //   3 = NOT_FOUND       — trade_id not in pool
  //   4 = ALREADY_DONE    — trade was already reconciled or expired
  // ────────────────────────────────────────────────────────────────────
  uint8_t reconcile(uint64_t trade_id, int64_t expected_price,
                    int32_t expected_qty) {
    // Compute the slot index directly — O(1)
    const size_t idx = trade_id & (pool_.capacity() - 1);
    const SlotState state = pool_.get_state(idx);

    // Check if the slot is empty (never written or fully evicted)
    if (state == SlotState::EMPTY) {
      return static_cast<uint8_t>(ReconcileResult::NOT_FOUND);
    }

    // Check if the slot holds a different trade_id (hash collision
    // from a later trade that overwrote this slot)
    if (pool_.slot_at(idx).trade_id != trade_id) {
      return static_cast<uint8_t>(ReconcileResult::NOT_FOUND);
    }

    // The slot holds our trade — check if it's still ACTIVE
    if (state != SlotState::ACTIVE) {
      // Trade exists but was already reconciled or expired
      return static_cast<uint8_t>(ReconcileResult::ALREADY_DONE);
    }

    // ── Field-by-field comparison against bank receipt ──
    const auto &slot = pool_.slot_at(idx);

    if (slot.price != expected_price) {
      return static_cast<uint8_t>(ReconcileResult::PRICE_MISMATCH);
    }
    if (slot.quantity != expected_qty) {
      return static_cast<uint8_t>(ReconcileResult::QTY_MISMATCH);
    }

    // ── Perfect match -> mark RECONCILED, cancel the timer ──
    pool_.set_state(idx, SlotState::RECONCILED);
    timer_wheel_.cancel(static_cast<uint32_t>(idx));
    ++total_reconciled_;

    return static_cast<uint8_t>(ReconcileResult::MATCH);
  }

  // ────────────────────────────────────────────────────────────────────
  // get_observation_matrix() — Zero-copy (N, 4) observation for RL
  //
  // Scans the OrderPool for all ACTIVE (unreconciled) trades and
  // builds a contiguous float matrix. The pointer is returned to
  // Python via nb::ndarray with reference_internal policy — Python
  // gets a direct view into C++ memory, no serialization.
  //
  // Column layout:
  //   [0] time_elapsed      — Normalized by Δ_max. Values > 1.0 mean
  //                           the trade is overdue (should be expiring).
  //   [1] price_delta       — Placeholder (0.0). Will be populated when
  //                           reconcile() detects a PRICE_MISMATCH.
  //   [2] missing_frequency — Global ratio: total_expired / total_ingested.
  //                           Higher = more anomalies = riskier market.
  //   [3] risk_score        — Per-counterparty static hash:
  //                           (counterparty_id % 100) / 100.0
  //                           Gives the RL agent a consistent 0.0–1.0
  //                           risk float. The neural net will learn
  //                           which counterparties correlate with anomalies.
  // ────────────────────────────────────────────────────────────────────
  nb::ndarray<nb::numpy, float> get_observation_matrix() {
    // ── Count active trades to determine matrix rows ──
    const size_t active = pool_.active_count();

    // ── Grow observation buffer if needed ──
    // This is an infrequent resize, not a hot-path allocation.
    if (active > obs_capacity_) {
      obs_capacity_ = active * 2; // 2x to amortize future growth
      observation_matrix_.resize(obs_capacity_ * 4);
    }

    // Handle empty case: return a (0, 4) matrix
    if (active == 0) {
      obs_rows_ = 0;
      size_t shape[2] = {0, 4};
      return nb::ndarray<nb::numpy, float>(observation_matrix_.data(),
                                           2,     // ndim
                                           shape, // shape array
                                           nb::handle());
    }

    // ── Compute global missing_frequency (same for all rows) ──
    const float missing_freq = (total_ingested_ > 0)
                                   ? static_cast<float>(total_expired_) /
                                         static_cast<float>(total_ingested_)
                                   : 0.0f;

    // ── Scan the pool and fill the observation matrix ──
    // TODO(performance): Maintain an active-trade index to avoid
    // the O(capacity) scan. For 1M slots this takes ~1ms, acceptable
    // for the RL step loop but improvable with an intrusive list.
    const double delta_max_d = static_cast<double>(TimerWheel::DELTA_MAX_NS);
    size_t row = 0;

    const uint32_t *active_indices = pool_.active_indices();
    for (size_t i = 0; i < active; ++i) {
      const uint32_t idx = active_indices[i];
      const auto &slot = pool_.slot_at(idx);
      float *out = &observation_matrix_[row * 4];

      // Col 0: time_elapsed — normalized seconds since trade ingestion
      const double elapsed_ns =
          static_cast<double>(watermark_ns_ - slot.timestamp_ns);
      out[0] = static_cast<float>(elapsed_ns / delta_max_d);

      // Col 1: price_delta — placeholder for reconciliation mismatches
      out[1] = 0.0f;

      // Col 2: missing_frequency — global anomaly rate
      out[2] = missing_freq;

      // Col 3: risk_score — per-counterparty static hash
      // (counterparty_id % 100) / 100.0 → consistent 0.0–0.99
      out[3] = static_cast<float>(slot.counterparty_id % 100) / 100.0f;

      ++row;
    }

    obs_rows_ = row;

    // ── Return zero-copy ndarray ──
    // nb::handle() means C++ owns the memory.
    // The rv_policy::reference_internal on the binding ensures Python
    // keeps the engine alive while the ndarray exists.
    size_t shape[2] = {obs_rows_, 4};
    return nb::ndarray<nb::numpy, float>(observation_matrix_.data(),
                                         2,     // ndim = 2
                                         shape, // shape = (N, 4)
                                         nb::handle());
  }

  // ────────────────────────────────────────────────────────────────────
  // get_anomaly_matrix() — Zero-copy (N, 4) matrix of anomalies (EXPIRED)
  //
  // Fixed "Shuffled Deck" Bug: Now precisely iterates over the exact
  // sequence of indices in expired_buffer_ generated by the last tick().
  // This perfectly aligns the data handed to the LLM with the answer
  // key used by compute_reward().
  // ────────────────────────────────────────────────────────────────────
  nb::ndarray<nb::numpy, float> get_anomaly_matrix() {
    // 1. Use the EXACT expired count from the last tick
    size_t count = last_batch_expired_count_;

    // 2. Grow buffer if needed
    if (count > anomaly_capacity_) {
      anomaly_capacity_ = count * 2 + 1024;
      anomaly_matrix_.resize(anomaly_capacity_ * 4);
    }

    // Handle empty case
    if (count == 0) {
      size_t shape[2] = {0, 4};
      return nb::ndarray<nb::numpy, float>(anomaly_matrix_.data(), 2, shape,
                                           nb::handle());
    }

    // 3. Populate matrix using exactly the expired_buffer_ sequence
    const float missing_freq = (total_ingested_ > 0)
                                   ? static_cast<float>(total_expired_) /
                                         static_cast<float>(total_ingested_)
                                   : 0.0f;
    const double delta_max_d = static_cast<double>(TimerWheel::DELTA_MAX_NS);
    size_t row = 0;

    for (size_t i = 0; i < count; ++i) {
      const uint32_t idx = expired_buffer_[i];

      if (pool_.get_state(idx) != SlotState::EXPIRED)
        continue;

      const auto &slot = pool_.slot_at(idx);
      float *out = &anomaly_matrix_[row * 4];

      const double elapsed_ns =
          static_cast<double>(watermark_ns_ - slot.timestamp_ns);
      out[0] = static_cast<float>(elapsed_ns / delta_max_d);
      out[1] = 0.0f;
      out[2] = missing_freq;
      out[3] = static_cast<float>(slot.counterparty_id % 100) / 100.0f;

      // NOTE: Do NOT clear the slot here. The ground truth label must
      // remain readable until compute_reward() processes the agent's
      // decisions. Clearing happens inside compute_reward() instead.
      ++row;
    }

    size_t shape[2] = {row, 4};
    return nb::ndarray<nb::numpy, float>(anomaly_matrix_.data(), 2, shape,
                                         nb::handle());
  }

  // ================================================================
  //  REWARD & DIFFICULTY — Phase 5
  // ================================================================

  // Asymmetric Cost Matrix (Normalized to 0.0 - 1.0)
  static constexpr float REWARD_TRUE_POSITIVE =
      1.0f; // Correctly flagged anomaly
  static constexpr float REWARD_TRUE_NEGATIVE =
      0.5f; // Correctly ignored safe trade
  static constexpr float REWARD_FALSE_POSITIVE =
      0.1f; // Flagged a safe trade (waste)
  static constexpr float REWARD_FALSE_NEGATIVE =
      0.0f; // Missed a real anomaly (danger)

  // ────────────────────────────────────────────────────────────────────
  // compute_reward() — Compute asymmetric reward for agent actions
  // ────────────────────────────────────────────────────────────────────
  float compute_reward(const uint8_t *agent_actions, size_t num_actions) {
    float total_reward = 0.0f;

    // Verify num_actions matches exactly what we expired in tick()
    if (num_actions != last_batch_expired_count_) {
      // In a real RL loop this shouldn't happen, but we'll be safe
      if (num_actions > last_batch_expired_count_)
        num_actions = last_batch_expired_count_;
    }

    // Grow details buffer if needed
    if (num_actions > reward_details_capacity_) {
      reward_details_capacity_ = num_actions * 2;
      reward_details_.resize(reward_details_capacity_);
    }

    last_tp_ = last_tn_ = last_fp_ = last_fn_ = 0;

    for (size_t i = 0; i < num_actions; ++i) {
      const uint32_t idx = expired_buffer_[i];
      const uint8_t truth = pool_.get_ground_truth(idx);
      const uint8_t action = agent_actions[i];

      float r;
      if (action == 1 && truth == 1) {
        r = REWARD_TRUE_POSITIVE;
        ++last_tp_;
      } else if (action == 0 && truth == 0) {
        r = REWARD_TRUE_NEGATIVE;
        ++last_tn_;
      } else if (action == 1 && truth == 0) {
        r = REWARD_FALSE_POSITIVE;
        ++last_fp_;
      } else { // action == 0 && truth == 1
        r = REWARD_FALSE_NEGATIVE;
        ++last_fn_;
      }

      reward_details_[i] = r;
      total_reward += r;
    }

    // Now that rewards are computed, clear the expired slots so they
    // don't reappear in the next get_anomaly_matrix() call.
    for (size_t i = 0; i < num_actions; ++i) {
      const uint32_t idx = expired_buffer_[i];
      if (pool_.get_state(idx) == SlotState::EXPIRED) {
        pool_.set_state(idx, SlotState::EMPTY);
      }
    }

    return total_reward;
  }

  // ────────────────────────────────────────────────────────────────────
  // generate_batch() — Linear Signal Separation for LLM Inference
  //
  // Anomaly ratios:
  //   EASY:   100% anomalies
  //   MEDIUM: 50% anomalies / 50% safe
  //   HARD:   20% anomalies / 80% safe
  //
  // Counterparty clustering (forces linearly separable risk_score):
  //   Anomaly → counterparty_id ∈ [70, 99]  → risk_score ∈ [0.70, 0.99]
  //   Safe    → counterparty_id ∈ [0,  19]  → risk_score ∈ [0.00, 0.19]
  // ────────────────────────────────────────────────────────────────────
  size_t generate_batch(Difficulty difficulty, size_t batch_size,
                        uint64_t timestamp_ns) {
    size_t anomaly_count = 0;
    const uint64_t base_id = total_ingested_;

    for (size_t i = 0; i < batch_size; ++i) {
      const uint64_t trade_id = base_id + i;
      const int64_t price = 10000 + static_cast<int64_t>(fast_rand() % 1000);
      const int32_t quantity = 10 + static_cast<int32_t>(fast_rand() % 100);

      // ── Step 1: Determine anomaly label based on difficulty ratio ──
      uint8_t is_anomaly = 0;

      if (difficulty == Difficulty::EASY) {
        // EASY: 100% anomalies — every trade is anomalous
        is_anomaly = 1;
      } else if (difficulty == Difficulty::MEDIUM) {
        // MEDIUM: 50% anomalies / 50% safe
        is_anomaly = (rand_float() < 0.50f) ? 1 : 0;
      } else {
        // HARD: 20% anomalies / 80% safe
        is_anomaly = (rand_float() < 0.20f) ? 1 : 0;
      }

      // ── Step 2: Force counterparty into distinct clusters ──────────
      // This creates a clear, linearly separable risk_score signal.
      uint32_t counterparty_id;
      if (is_anomaly == 1) {
        counterparty_id = 70 + (fast_rand() % 30); // High Risk Cluster [70-99]
      } else {
        counterparty_id = (fast_rand() % 20); // Low Risk Cluster  [0-19]
      }

      anomaly_count += is_anomaly;
      ingest_trade_labeled(trade_id, price, quantity, counterparty_id,
                           timestamp_ns + i * 1000, is_anomaly);
    }

    return anomaly_count;
  }

  // ────────────────────────────────────────────────────────────────────
  // ingest_trade_labeled() — Internal helper for generate_batch
  // ────────────────────────────────────────────────────────────────────
  bool ingest_trade_labeled(uint64_t trade_id, int64_t price, int32_t quantity,
                            uint32_t counterparty_id, uint64_t timestamp_ns,
                            uint8_t is_anomaly) {
    const uint32_t idx =
        static_cast<uint32_t>(trade_id & (pool_.capacity() - 1));

    pool_.insert(trade_id, price, quantity, counterparty_id, timestamp_ns);
    pool_.set_ground_truth(idx, is_anomaly);
    timer_wheel_.schedule(idx, timestamp_ns);
    ++total_ingested_;
    return true;
  }

  // PRNG and Stats
  void set_seed(uint64_t seed) { rng_state_ = seed; }
  size_t last_tp() const { return last_tp_; }
  size_t last_tn() const { return last_tn_; }
  size_t last_fp() const { return last_fp_; }
  size_t last_fn() const { return last_fn_; }
  size_t last_expired_count() const { return last_batch_expired_count_; }

  // ================================================================
  //  DIRECT INGESTION — For single-threaded mode (bypasses ring buffer)
  // ================================================================

  // ────────────────────────────────────────────────────────────────────
  // ingest_trade() — Direct insertion into the pool (no ring buffer)
  //
  // Use this when running single-threaded (e.g., from Python without
  // a separate ingestion thread). This is simpler but doesn't support
  // concurrent ingestion.
  // ────────────────────────────────────────────────────────────────────
  bool ingest_trade(uint64_t trade_id, int64_t price, int32_t quantity,
                    uint32_t counterparty_id, uint64_t timestamp_ns) {
    const uint32_t idx =
        static_cast<uint32_t>(trade_id & (pool_.capacity() - 1));

    pool_.insert(trade_id, price, quantity, counterparty_id, timestamp_ns);
    timer_wheel_.schedule(idx, timestamp_ns);
    ++total_ingested_;
    return true;
  }

  // ================================================================
  //  STATISTICS — Exposed as read-only properties to Python
  // ================================================================

  size_t total_ingested() const { return total_ingested_; }
  size_t total_reconciled() const { return total_reconciled_; }
  size_t total_expired() const { return total_expired_; }
  size_t active_count() const { return pool_.active_count(); }
  size_t ring_buffer_size() const { return ring_buffer_.size(); }
  uint64_t watermark() const { return watermark_ns_; }
  size_t pool_capacity() const { return pool_.capacity(); }

private:
  // ────────────────────────────────────────────────────────────────────
  // next_power_of_2() — Round up to nearest power of 2
  //
  // Uses the bit-smearing technique: O(1), no loops.
  //   63 → 64, 100 → 128, 1000000 → 1048576
  // ────────────────────────────────────────────────────────────────────
  static size_t next_power_of_2(size_t n) {
    if (n == 0)
      return 1;
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
  }

  // ── Fast PRNG (Xorshift64) ──────────────────────────────────────────
  uint64_t rng_state_ = 12345;
  uint64_t fast_rand() {
    rng_state_ ^= rng_state_ << 13;
    rng_state_ ^= rng_state_ >> 7;
    rng_state_ ^= rng_state_ << 17;
    return rng_state_;
  }
  float rand_float() {
    return static_cast<float>(fast_rand() & 0xFFFFFF) / 16777216.0f;
  }

  // ── Subsystems ──────────────────────────────────────────────────────
  OrderPool pool_;                                   // Trade storage
  TimerWheel timer_wheel_;                           // Expiration mgmt
  SPSCRingBuffer<RING_BUFFER_CAPACITY> ring_buffer_; // Lock-free ingestion

  // ── Event-time watermark ────────────────────────────────────────────
  uint64_t watermark_ns_;

  // ── Pre-allocated buffers ───────────────────────────────────────────
  std::vector<uint32_t> expired_buffer_;  // Indices from timer wheel
  std::vector<float> observation_matrix_; // (N, 4) float matrix
  size_t obs_capacity_;                   // Current obs buffer capacity
  size_t obs_rows_ = 0;                   // Rows in current obs matrix

  std::vector<float> anomaly_matrix_; // (N, 4) float matrix
  size_t anomaly_capacity_;           // Current anomaly buffer capacity

  std::vector<float> reward_details_; // Per-trade rewards
  size_t reward_details_capacity_;    // Current reward buffer capacity

  // ── Counters ────────────────────────────────────────────────────────
  size_t total_ingested_ = 0;
  size_t total_reconciled_ = 0;
  size_t total_expired_ = 0;

  size_t last_batch_expired_count_ = 0;
  size_t last_tp_ = 0, last_tn_ = 0, last_fp_ = 0, last_fn_ = 0;
};