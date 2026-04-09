#pragma once
// ============================================================================
// TimerWheel — O(1) Hierarchical Timer Wheel for Trade Expiration
// ============================================================================
//
// Refactored to follow Phase 3 Task Execution Plan:
//   - Zero-allocation (pre-allocated vectors)
//   - Intrusive linked-lists using uint32_t indices (no pointers)
//   - 3-Level hierarchy for O(1) scheduling and expiration
//
// ============================================================================

#include <cstdint>
#include <vector>

class TimerWheel {
public:
    // ── 1. Constant Definitions ──────────────────────────────────────────
    static constexpr uint64_t L0_GRANULARITY_NS = 100'000ULL;              // 100 us
    static constexpr uint64_t L1_GRANULARITY_NS = 256ULL * 100'000ULL;     // 25.6 ms
    static constexpr uint64_t L2_GRANULARITY_NS = 256ULL * 256ULL * 100'000ULL; // 6.5536 s

    static constexpr size_t   SLOTS_PER_LEVEL   = 256;
    static constexpr uint32_t TIMER_NULL        = 0xFFFFFFFF;

    // Δ_max = 5.0 seconds
    static constexpr uint64_t DELTA_MAX_NS       = 5'000'000'000ULL;

    // ── Constructor ──────────────────────────────────────────────────────
    explicit TimerWheel(size_t capacity)
        : watermark_ns_(0)
        , L0_idx_(0)
        , L1_idx_(0)
        , L2_idx_(0)
    {
        // 2. Declaration & Initialization of bucket heads
        head_L0_.resize(SLOTS_PER_LEVEL, TIMER_NULL);
        head_L1_.resize(SLOTS_PER_LEVEL, TIMER_NULL);
        head_L2_.resize(SLOTS_PER_LEVEL, TIMER_NULL);

        // 3. Monolithic next_ptr array matching OrderPool Capacity
        next_ptr_.resize(capacity, TIMER_NULL);
        
        // Storage for actual expiry times (required for verification in Task 6)
        expiry_times_.resize(capacity, 0);
    }

    // ── 5. schedule() — O(1) Insertion ──────────────────────────────────
    void schedule(uint32_t slot_idx, uint64_t trade_timestamp) {
        const uint64_t expiry = trade_timestamp + DELTA_MAX_NS;
        expiry_times_[slot_idx] = expiry;

        // Compute delta from current watermark_ns_ for level selection
        const uint64_t delta = (expiry > watermark_ns_) ? (expiry - watermark_ns_) : 0;

        // Hash the delta into the correct level (L0, L1, or L2)
        if (delta < SLOTS_PER_LEVEL * L0_GRANULARITY_NS) {
            // Falls into L0
            const uint32_t slot = static_cast<uint32_t>((expiry / L0_GRANULARITY_NS) & 0xFF);
            next_ptr_[slot_idx] = head_L0_[slot];
            head_L0_[slot]      = slot_idx;
        } else if (delta < SLOTS_PER_LEVEL * L1_GRANULARITY_NS) {
            // Falls into L1
            const uint32_t slot = static_cast<uint32_t>((expiry / L1_GRANULARITY_NS) & 0xFF);
            next_ptr_[slot_idx] = head_L1_[slot];
            head_L1_[slot]      = slot_idx;
        } else {
            // Falls into L2
            const uint32_t slot = static_cast<uint32_t>((expiry / L2_GRANULARITY_NS) & 0xFF);
            next_ptr_[slot_idx] = head_L2_[slot];
            head_L2_[slot]      = slot_idx;
        }
    }

    // ── Lazy cancel() — O(1) ───────────────────────────────────────────
    void cancel(uint32_t slot_idx) {
        // We rely on lazy-cancellation by clearing the timestamp.
        // The advance() sweep will skip any nodes with 0 timestamps.
        expiry_times_[slot_idx] = 0;
    }

    // ── 6. advance() — Event-Time Expiration Logic ────────────────────
    size_t advance(uint64_t new_watermark_ns, uint32_t* expired_out, size_t max_out) {
        if (new_watermark_ns <= watermark_ns_) return 0;

        size_t count = 0;
        // Compute how many L0 buckets need passing
        const uint64_t ticks = (new_watermark_ns - watermark_ns_) / L0_GRANULARITY_NS;

        // Tight loop over elapsed L0 ticks
        for (uint64_t i = 0; i < ticks && count < max_out; ++i) {
            // Increment watermark L0 step by step for cascading logic
            watermark_ns_ += L0_GRANULARITY_NS;
            
            // Current L0 slot index
            const uint32_t slot_l0 = static_cast<uint32_t>((watermark_ns_ / L0_GRANULARITY_NS) & 0xFF);
            L0_idx_ = slot_l0;

            // ── Check for Revolutions (Cascading) ──
            if (slot_l0 == 0) {
                // L0 completed a revolution. Cascade L1 -> L0
                L1_idx_ = (L1_idx_ + 1) & 0xFF;
                cascade_level(1, L1_idx_, 0);
                
                if (L1_idx_ == 0) {
                    // L1 completed a revolution. Cascade L2 -> L1
                    L2_idx_ = (L2_idx_ + 1) & 0xFF;
                    cascade_level(2, L2_idx_, 1);
                }
            }

            // ── Sweep L0 Bucket ──
            uint32_t curr = head_L0_[slot_l0];
            head_L0_[slot_l0] = TIMER_NULL; // Detach list

            while (curr != TIMER_NULL && count < max_out) {
                const uint32_t next = next_ptr_[curr];
                const uint64_t expiry = expiry_times_[curr];

                if (expiry != 0) { // Not lazily cancelled
                    if (expiry <= new_watermark_ns) {
                        // EXPIRED: report trade index
                        expired_out[count++] = curr;
                    } else {
                        // Re-insert into L0 (handled after cascading logic for accuracy)
                        const uint32_t re_slot = static_cast<uint32_t>((expiry / L0_GRANULARITY_NS) & 0xFF);
                        next_ptr_[curr] = head_L0_[re_slot];
                        head_L0_[re_slot] = curr;
                    }
                }
                curr = next;
            }
        }

        // Catch up any remaining rounding delta
        watermark_ns_ = new_watermark_ns;
        return count;
    }

private:
    // Helper to redistribute entries from a higher bucket into a lower level
    void cascade_level(int from_level, uint32_t slot, int to_level) {
        uint32_t curr = TIMER_NULL;
        
        // Detach the list from the target bucket
        if (from_level == 1) {
            curr = head_L1_[slot];
            head_L1_[slot] = TIMER_NULL;
        } else if (from_level == 2) {
            curr = head_L2_[slot];
            head_L2_[slot] = TIMER_NULL;
        }

        while (curr != TIMER_NULL) {
            const uint32_t next = next_ptr_[curr];
            const uint64_t expiry = expiry_times_[curr];

            if (expiry != 0) {
                // Re-hash into the specified level (0 or 1)
                if (to_level == 0) {
                    const uint32_t target_slot = static_cast<uint32_t>((expiry / L0_GRANULARITY_NS) & 0xFF);
                    next_ptr_[curr] = head_L0_[target_slot];
                    head_L0_[target_slot] = curr;
                } else if (to_level == 1) {
                    const uint32_t target_slot = static_cast<uint32_t>((expiry / L1_GRANULARITY_NS) & 0xFF);
                    next_ptr_[curr] = head_L1_[target_slot];
                    head_L1_[target_slot] = curr;
                }
            }
            curr = next;
        }
    }

    // ── Subsystems ──────────────────────────────────────────────────────
    std::vector<uint32_t> head_L0_;     // L0 Wheel heads
    std::vector<uint32_t> head_L1_;     // L1 Wheel heads
    std::vector<uint32_t> head_L2_;     // L2 Wheel heads

    // Monolithic intrusive next array (1:1 with OrderPool indices)
    std::vector<uint32_t> next_ptr_;

    // Expiration timestamp storage (zero-allocation pool)
    std::vector<uint64_t> expiry_times_;

    // ── State Variables ──────────────────────────────────────────────────
    uint64_t watermark_ns_;

    // Current index positions for each level
    uint32_t L0_idx_;
    uint32_t L1_idx_;
    uint32_t L2_idx_;
};
