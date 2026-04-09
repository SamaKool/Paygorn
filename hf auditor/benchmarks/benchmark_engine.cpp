#include "reconciliation_engine.hpp"
#include <benchmark/benchmark.h>
#include <cstdint>
#include <memory>

// ============================================================================
// Benchmark 1: BM_RingBuffer_PushPop
// Purpose: Measure pure SPSC ring buffer push + drain throughput.
// ============================================================================
class RingBufferFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& /*state*/) override {
        // PRE-ALLOCATION: Construct engine in setup, not the hot loop
        engine_ = std::make_unique<ReconciliationEngine>(1 << 20);
    }

    void TearDown(const ::benchmark::State& /*state*/) override {
        engine_.reset();
    }

    std::unique_ptr<ReconciliationEngine> engine_;
};

BENCHMARK_DEFINE_F(RingBufferFixture, BM_RingBuffer_PushPop)(benchmark::State& state) {
    uint64_t trade_id = 0;
    
    for (auto _ : state) {
        // 1. Submit trade (Producer Side)
        bool pushed = engine_->submit_trade(trade_id, 10000, 50, 42, trade_id * 1000);
        benchmark::DoNotOptimize(pushed);
        
        // 2. Tick (Consumer Side - drains ring buffer)
        size_t drained = engine_->tick(0);
        benchmark::DoNotOptimize(drained);
        
        // 3. Compiler Fence
        benchmark::ClobberMemory();
        
        ++trade_id;
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(RingBufferFixture, BM_RingBuffer_PushPop)->Threads(1);

// ============================================================================
// Benchmark 2: BM_Engine_Reconcile
// Purpose: Measure O(1) trade lookup + field comparison speed.
// ============================================================================
class ReconcileFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& /*state*/) override {
        engine_ = std::make_unique<ReconciliationEngine>(1 << 20);
        
        // PRE-POPULATE: Load with trades before benchmarking the lookup
        for (uint64_t i = 0; i < 65536; ++i) {
            bool ingested = engine_->ingest_trade(i, 10000, 50, 42, 1000);
            benchmark::DoNotOptimize(ingested);
        }
        benchmark::ClobberMemory();
    }

    void TearDown(const ::benchmark::State& /*state*/) override {
        engine_.reset();
    }

    std::unique_ptr<ReconciliationEngine> engine_;
};

BENCHMARK_DEFINE_F(ReconcileFixture, BM_Engine_Reconcile)(benchmark::State& state) {
    uint64_t trade_id = 0;
    const uint64_t mask = 65535;
    
    for (auto _ : state) {
        // Benchmark the O(1) lookup speed
        uint8_t result = engine_->reconcile(trade_id & mask, 10000, 50);
        benchmark::DoNotOptimize(result);
        
        benchmark::ClobberMemory();
        ++trade_id;
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(ReconcileFixture, BM_Engine_Reconcile)->Threads(1);

// ============================================================================
// Benchmark 3: BM_Engine_Tick_Anomalies
// Purpose: Measure Timer Wheel expiration sweep speed under load.
// ============================================================================
class TickAnomalyFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& /*state*/) override {
        engine_ = std::make_unique<ReconciliationEngine>(1 << 20);
        
        // PRE-POPULATE: Ingest trades at time 0
        for (uint64_t i = 0; i < 65536; ++i) {
            bool ingested = engine_->ingest_trade(i, 10000, 50, i % 100, 0);
            benchmark::DoNotOptimize(ingested);
        }
        
        // Run tick to place them in the timer wheel slots
        size_t initial_tick = engine_->tick(0);
        benchmark::DoNotOptimize(initial_tick);
        benchmark::ClobberMemory();
    }

    void TearDown(const ::benchmark::State& /*state*/) override {
        engine_.reset();
    }

    std::unique_ptr<ReconciliationEngine> engine_;
};

BENCHMARK_DEFINE_F(TickAnomalyFixture, BM_Engine_Tick_Anomalies)(benchmark::State& state) {
    // Past the 5-second delta_max
    uint64_t watermark = 6'000'000'000ULL;
    
    for (auto _ : state) {
        // Benchmark the Timer Wheel sweep + eviction logic
        size_t ingested = engine_->tick(watermark);
        benchmark::DoNotOptimize(ingested);
        
        benchmark::ClobberMemory();
        watermark += 1'000'000'000ULL; // Increment to stay monotonic
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(TickAnomalyFixture, BM_Engine_Tick_Anomalies)->Threads(1);
