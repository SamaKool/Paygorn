// ============================================================================
// auditor.cpp — Nanobind Python Bindings for the ReconciliationEngine
// ============================================================================
//
// This file is the SOLE bridge between C++ and Python. It contains
// ONLY binding definitions — no business logic. All engine code lives
// in the header files:
//   - order_pool.hpp           → O(1) flat-array trade storage
//   - timer_wheel.hpp          → O(1) hierarchical timer wheel
//   - spsc_ring_buffer.hpp     → Lock-free SPSC ring buffer
//   - reconciliation_engine.hpp → Orchestrator
//
// The compiled .pyd module exposes a single class:
//   hft_auditor.ReconciliationEngine
//
// ============================================================================

#include "reconciliation_engine.hpp"

NB_MODULE(hft_auditor, m) {
  nb::enum_<Difficulty>(m, "Difficulty")
      .value("EASY",   Difficulty::EASY)
      .value("MEDIUM", Difficulty::MEDIUM)
      .value("HARD",   Difficulty::HARD)
      .export_values();

  m.doc() = "High-Frequency Trading Reconciliation Engine\n"
            "C++20 zero-allocation backend for RL-based trade auditing.\n"
            "Processes 1M+ trades/sec with O(1) matching, O(1) expiration,\n"
            "and lock-free concurrent ingestion.";

  nb::class_<ReconciliationEngine>(m, "ReconciliationEngine")

      // ── Constructor ──────────────────────────────────────────────
      .def(nb::init<size_t>(), nb::arg("capacity") = 1048576,
           "Initialize the engine with a pre-allocated trade pool.\n"
           "Capacity is rounded up to the next power of 2.\n"
           "Default: 1,048,576 (2^20) slots.")

      // ── Producer API (lock-free, thread-safe) ────────────────────
      .def("submit_trade", &ReconciliationEngine::submit_trade,
           nb::arg("trade_id"), nb::arg("price"), nb::arg("quantity"),
           nb::arg("counterparty_id"), nb::arg("timestamp_ns"),
           "Enqueue a trade into the lock-free SPSC ring buffer.\n"
           "Safe to call from a separate ingestion thread.\n"
           "Returns False if the buffer is full (back-pressure).")

      // ── Engine API (single-threaded) ─────────────────────────────
      .def("tick", &ReconciliationEngine::tick, nb::arg("new_watermark_ns"),
           "Main processing loop. Call once per RL step.\n"
           "1. Drains the SPSC ring buffer into the OrderPool\n"
           "2. Advances the timer wheel watermark\n"
           "3. Expires trades older than Δ_max (5.0 seconds)\n"
           "Returns: number of trades ingested from the ring buffer.")

      .def("reconcile", &ReconciliationEngine::reconcile, nb::arg("trade_id"),
           nb::arg("expected_price"), nb::arg("expected_qty"),
           "Match a trade against an external bank receipt.\n"
           "Returns: 0=MATCH, 1=PRICE_MISMATCH, 2=QTY_MISMATCH,\n"
           "         3=NOT_FOUND, 4=ALREADY_DONE")

      .def("ingest_trade", &ReconciliationEngine::ingest_trade,
           nb::arg("trade_id"), nb::arg("price"), nb::arg("quantity"),
           nb::arg("counterparty_id"), nb::arg("timestamp_ns"),
           "Direct insertion bypassing the ring buffer.\n"
           "Use in single-threaded mode (no separate ingestion thread).")

      .def("ingest_trade_labeled", &ReconciliationEngine::ingest_trade_labeled,
           nb::arg("trade_id"), nb::arg("price"), nb::arg("quantity"),
           nb::arg("counterparty_id"), nb::arg("timestamp_ns"),
           nb::arg("is_anomaly"),
           "Ingest a trade with a known ground-truth label (0=safe, 1=anomaly).")

      .def("generate_batch", &ReconciliationEngine::generate_batch,
           nb::arg("difficulty"), nb::arg("batch_size"), nb::arg("timestamp_ns"),
           "Generate a batch of trades with difficulty-appropriate anomaly labels.\n"
           "Returns: number of anomalies in the batch.")

      .def("compute_reward",
           [](ReconciliationEngine& self, nb::ndarray<uint8_t, nb::ndim<1>> actions) -> float {
               return self.compute_reward(actions.data(), actions.shape(0));
           },
           nb::arg("agent_actions"),
           "Compute asymmetric reward for the agent's decisions on expired trades.\n"
           "Cost matrix: TP=+1.0, TN=+0.5, FP=+0.1, FN=0.0.")

      .def("set_seed", &ReconciliationEngine::set_seed, nb::arg("seed"),
           "Set the PRNG seed for reproducible batch generation.")

      // ── Observation (zero-copy) ──────────────────────────────────
      .def("get_observation_matrix",
           &ReconciliationEngine::get_observation_matrix,
           nb::rv_policy::reference_internal,
           "Zero-copy (N, 4) numpy matrix of active trade features:\n"
           "  [0] time_elapsed:      (watermark - trade_ts) / Δ_max\n"
           "  [1] price_delta:       placeholder (0.0)\n"
           "  [2] missing_frequency: total_expired / total_ingested\n"
           "  [3] risk_score:        (counterparty_id %% 100) / 100.0\n"
           "Memory is owned by C++; Python must not outlive the engine.")

      .def("get_anomaly_matrix",
             &ReconciliationEngine::get_anomaly_matrix,
             nb::rv_policy::reference_internal,
             "Zero-copy (N, 4) numpy matrix of EXPIRED trade features.\n"
             "Memory owned by C++; do not retain reference beyond next tick().")

      // ── Read-only properties ─────────────────────────────────────
      .def_prop_ro("total_ingested", &ReconciliationEngine::total_ingested,
                   "Total trades ingested since engine creation.")
      .def_prop_ro("total_reconciled", &ReconciliationEngine::total_reconciled,
                   "Total trades successfully reconciled.")
      .def_prop_ro("total_expired", &ReconciliationEngine::total_expired,
                   "Total trades expired (exceeded Δ_max threshold).")
      .def_prop_ro("last_expired_count", &ReconciliationEngine::last_expired_count,
                   "Number of trades expired in the most recent tick() call.")
      .def_prop_ro("last_tp", &ReconciliationEngine::last_tp, "True Positives in last reward call.")
      .def_prop_ro("last_tn", &ReconciliationEngine::last_tn, "True Negatives in last reward call.")
      .def_prop_ro("last_fp", &ReconciliationEngine::last_fp, "False Positives in last reward call.")
      .def_prop_ro("last_fn", &ReconciliationEngine::last_fn, "False Negatives in last reward call.")
      .def_prop_ro("active_count", &ReconciliationEngine::active_count,
                   "Number of currently active (unreconciled) trades.")
      .def_prop_ro("ring_buffer_size", &ReconciliationEngine::ring_buffer_size,
                   "Current number of entries in the SPSC ring buffer.")
      .def_prop_ro("watermark", &ReconciliationEngine::watermark,
                   "Current event-time watermark in nanoseconds.")
      .def_prop_ro("pool_capacity", &ReconciliationEngine::pool_capacity,
                   "Total pool capacity (power of 2).");
}