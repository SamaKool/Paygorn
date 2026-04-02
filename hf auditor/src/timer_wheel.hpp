#pragma once
// ============================================================================
// TimerWheel — O(1) Hierarchical Timer Wheel for Trade Expiration
// ============================================================================
//
// CONCEPT:
//   A 3-level timer wheel inspired by the Linux kernel's timer infrastructure.
//   When a trade is ingested, it's scheduled to expire at:
//       expiry = insertion_timestamp + Δ_max (5.0 seconds)
//
//   As the global event-time watermark advances, the wheel sweeps expired
//   slots and cascades entries from higher levels to lower levels.
//
// LEVEL PARAMETERS:
//   ┌───────┬───────┬──────────────┬──────────────┐
//   │ Level │ Slots │ Granularity  │ Total Span   │
//   ├───────┼───────┼──────────────┼──────────────┤
//   │ L0    │ 256   │ 100 μs       │ 25.6 ms      │
//   │ L1    │ 256   │ 25.6 ms      │ 6.5536 s     │
//   │ L2    │ 256   │ 6.5536 s     │ 1677 s       │
//   └───────┴───────┴──────────────┴──────────────┘
//
//   L1 spans 6.55 seconds > Δ_max (5.0s), so newly scheduled trades land
//   in L1 and cascade to L0 as they approach expiry. L2 provides headroom
//   for delayed processing.
//
// COMPLEXITY:
//   - schedule():  O(1) — single level/slot computation + list prepend
//   - advance():   O(1) amortized — sweeps L0 ticks, cascades on wrap
//   - cancel():    O(1) — lazy cancel via sentinel
//
// EVENT-TIME ONLY:
//   Wall-clock time is NEVER used. The watermark is driven by the data's
//   embedded timestamps, as required by the architectural rules.
//
// ZERO-ALLOCATION GUARANTEE:
//   All TimerNode storage is pre-allocated in a flat std::vector at startup.
//   Linked lists are index-based (uint32_t indices, not pointers).
//
// ============================================================================

#include <cstdint>
#include <vector>
#include <array>
#include <cassert>

// ────────────────────────────────────────────────────────────────────────────
// Sentinel: "null pointer" for index-based linked lists
// ────────────────────────────────────────────────────────────────────────────
static constexpr uint32_t TIMER_NULL = UINT32_MAX;

// ────────────────────────────────────────────────────────────────────────────
// TimerNode — Intrusive linked-list element, 32-byte aligned
// ────────────────────────────────────────────────────────────────────────────
//
// Each node is pre-allocated in a flat pool indexed by the same index as the
// OrderPool slot. This gives us O(1) mapping between trades and their timers.
//
// Layout (total = 32 bytes):
//   [0..3]   pool_index       uint32_t    Back-pointer into OrderPool
//   [4..7]   next             uint32_t    Next node in this wheel slot's list
//   [8..15]  expiry_time_ns   uint64_t    Absolute expiry timestamp
//   [16]     wheel_level      uint8_t     Which level (0, 1, 2) this node lives in
//   [17]     wheel_slot       uint8_t     Which slot within the level (0..255)
//   [18..31] padding          14 bytes    Pad to 32-byte alignment
//
struct alignas(32) TimerNode {
    uint32_t pool_index;        // 4 bytes  — maps back to OrderPool[pool_index]
    uint32_t next;              // 4 bytes  — intrusive list: next node index
    uint64_t expiry_time_ns;    // 8 bytes  — 0 = cancelled (lazy sentinel)
    uint8_t  wheel_level;       // 1 byte   — current level in the hierarchy
    uint8_t  wheel_slot;        // 1 byte   — current slot within the level
    uint8_t  padding[14];       // 14 bytes — pad to 32
};
static_assert(sizeof(TimerNode) == 32, "TimerNode must be exactly 32 bytes");

// ────────────────────────────────────────────────────────────────────────────
// TimerWheel — 3-level hierarchical timer wheel
// ────────────────────────────────────────────────────────────────────────────
class TimerWheel {
public:
    // ── Level parameters ─────────────────────────────────────────────────
    //
    // L0 granularity of 100μs means a 1ms watermark advance only ticks
    // 10 L0 slots — fast inner loop. At 1M trades/sec with 1000-trade
    // batches, each batch spans ~1ms = 10 ticks.
    //
    static constexpr uint64_t L0_GRANULARITY_NS = 100'000ULL;              // 100 μs
    static constexpr uint64_t L1_GRANULARITY_NS = 256ULL * 100'000ULL;     // 25.6 ms
    static constexpr uint64_t L2_GRANULARITY_NS = 256ULL * 256ULL * 100'000ULL; // 6.5536 s

    static constexpr size_t   SLOTS_PER_LEVEL   = 256;
    static constexpr size_t   NUM_LEVELS         = 3;

    // Δ_max = 5.0 seconds — any unmatched trade older than this is an anomaly
    static constexpr uint64_t DELTA_MAX_NS       = 5'000'000'000ULL;

    // Maximum total span of the wheel (all 3 levels)
    static constexpr uint64_t TOTAL_SPAN_NS      = SLOTS_PER_LEVEL * L2_GRANULARITY_NS;

    // ────────────────────────────────────────────────────────────────────
    // Constructor: pre-allocate the full timer node pool
    // ────────────────────────────────────────────────────────────────────
    explicit TimerWheel(size_t node_capacity)
        : node_capacity_(node_capacity)
    {
        // Pre-allocate all timer nodes at startup. Each node corresponds
        // 1:1 with an OrderPool slot by index.
        nodes_.resize(node_capacity);

        // Initialize all wheel slots to empty (TIMER_NULL sentinel)
        for (auto& level : wheel_) {
            level.fill(TIMER_NULL);
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // schedule() — O(1) — Schedule a trade for expiration
    //
    // Called when a trade is ingested into the OrderPool. Computes the
    // expiry timestamp and inserts the timer node into the appropriate
    // wheel level and slot.
    //
    // pool_index: index into the OrderPool (same index for the TimerNode)
    // insertion_time_ns: the trade's event-time timestamp
    // ────────────────────────────────────────────────────────────────────
    void schedule(uint32_t pool_index, uint64_t insertion_time_ns) {
        const uint64_t expiry = insertion_time_ns + DELTA_MAX_NS;

        auto& node = nodes_[pool_index];
        node.pool_index     = pool_index;
        node.expiry_time_ns = expiry;

        // Determine which level and slot this expiry falls into.
        // We compute the delta from the current watermark to decide
        // the appropriate level (closer = lower level = finer granularity).
        const uint64_t delta = (expiry > watermark_ns_) 
                             ? (expiry - watermark_ns_) 
                             : 0ULL;

        uint8_t level;
        uint8_t slot;

        if (delta < SLOTS_PER_LEVEL * L0_GRANULARITY_NS) {
            // Within L0 range (< 25.6 ms from now)
            level = 0;
            slot  = static_cast<uint8_t>((expiry / L0_GRANULARITY_NS) & 0xFF);
        } else if (delta < SLOTS_PER_LEVEL * L1_GRANULARITY_NS) {
            // Within L1 range (< 6.55 s from now) — most trades land here
            level = 1;
            slot  = static_cast<uint8_t>((expiry / L1_GRANULARITY_NS) & 0xFF);
        } else {
            // Beyond L1 range — goes to L2
            level = 2;
            slot  = static_cast<uint8_t>((expiry / L2_GRANULARITY_NS) & 0xFF);
        }

        node.wheel_level = level;
        node.wheel_slot  = slot;

        // Prepend to the slot's intrusive linked list — O(1)
        node.next               = wheel_[level][slot];
        wheel_[level][slot]     = pool_index;
    }

    // ────────────────────────────────────────────────────────────────────
    // cancel() — O(1) — Cancel a timer (trade reconciled before expiry)
    //
    // Uses lazy cancellation: sets the expiry to 0 (sentinel). The
    // sweep loop will skip cancelled nodes. This avoids the expensive
    // O(N) unlink operation on the intrusive list.
    // ────────────────────────────────────────────────────────────────────
    void cancel(uint32_t pool_index) {
        nodes_[pool_index].expiry_time_ns = 0;  // Sentinel: "cancelled"
    }

    // ────────────────────────────────────────────────────────────────────
    // advance() — O(1) amortized — Advance the watermark, collect expired
    //
    // Sweeps L0 slots from the old watermark position to the new one.
    // When L0 wraps around (every 25.6 ms), cascades entries from L1→L0.
    // When L1 wraps around (every 6.55 s), cascades entries from L2→L1.
    //
    // expired_out: pre-allocated buffer to receive expired pool indices
    // max_expired: capacity of expired_out
    // Returns:     number of expired trades written to expired_out
    // ────────────────────────────────────────────────────────────────────
    size_t advance(uint64_t new_watermark_ns,
                   uint32_t* expired_out,
                   size_t max_expired)
    {
        // Guard: if watermark didn't advance, nothing to do
        if (new_watermark_ns <= watermark_ns_) {
            return 0;
        }

        size_t expired_count = 0;

        // Compute L0 tick range to sweep
        const uint64_t old_l0_tick = watermark_ns_ / L0_GRANULARITY_NS;
        const uint64_t new_l0_tick = new_watermark_ns / L0_GRANULARITY_NS;

        // ── Fast path: watermark jumped beyond the entire wheel span ──
        // This can happen on initial startup or after long idle periods.
        // In this case, all scheduled timers are definitely expired.
        if (new_watermark_ns - watermark_ns_ > TOTAL_SPAN_NS) {
            expired_count = flush_all_levels(expired_out, max_expired,
                                             new_watermark_ns);
            watermark_ns_ = new_watermark_ns;
            return expired_count;
        }

        // ── Normal path: sweep L0 ticks one by one ──
        // For each L0 tick, check if we need to cascade from higher levels.
        for (uint64_t tick = old_l0_tick + 1;
             tick <= new_l0_tick && expired_count < max_expired;
             ++tick)
        {
            const uint8_t l0_slot = static_cast<uint8_t>(tick & 0xFF);

            // Check if L0 just wrapped (every 256 ticks = 25.6 ms)
            if (l0_slot == 0) {
                const uint64_t l1_tick = tick / SLOTS_PER_LEVEL;
                const uint8_t  l1_slot = static_cast<uint8_t>(l1_tick & 0xFF);

                // Check if L1 also wrapped (every 65,536 ticks = 6.55 s)
                if (l1_slot == 0) {
                    const uint64_t l2_tick = l1_tick / SLOTS_PER_LEVEL;
                    const uint8_t  l2_slot = static_cast<uint8_t>(l2_tick & 0xFF);
                    // Cascade L2 → L1 first (higher level resolves first)
                    cascade_slot(2, l2_slot, 1);
                }

                // Cascade L1 → L0
                cascade_slot(1, l1_slot, 0);
            }

            // Sweep this L0 slot — collect expired entries
            expired_count += sweep_slot(0, l0_slot, new_watermark_ns,
                                        expired_out + expired_count,
                                        max_expired - expired_count);
        }

        watermark_ns_ = new_watermark_ns;
        return expired_count;
    }

    // ────────────────────────────────────────────────────────────────────
    // Accessors
    // ────────────────────────────────────────────────────────────────────
    uint64_t watermark() const { return watermark_ns_; }

    void set_initial_watermark(uint64_t wm) { watermark_ns_ = wm; }

private:
    // ────────────────────────────────────────────────────────────────────
    // cascade_slot() — Move entries from a higher-level slot to the
    //                  next lower level
    //
    // This is the core of the hierarchical design. When L0 wraps around,
    // we take one L1 slot and redistribute its entries into L0 slots
    // based on their expiry times at L0 granularity.
    // ────────────────────────────────────────────────────────────────────
    void cascade_slot(uint8_t from_level, uint8_t from_slot, uint8_t to_level) {
        // Detach the entire list from the source slot
        uint32_t node_idx       = wheel_[from_level][from_slot];
        wheel_[from_level][from_slot] = TIMER_NULL;

        // Determine the granularity of the target level
        const uint64_t to_granularity = (to_level == 0) ? L0_GRANULARITY_NS
                                                        : L1_GRANULARITY_NS;

        // Walk the list and re-insert each node into the target level
        while (node_idx != TIMER_NULL) {
            const uint32_t next = nodes_[node_idx].next;

            // Skip cancelled nodes (lazy cleanup)
            if (nodes_[node_idx].expiry_time_ns == 0) {
                node_idx = next;
                continue;
            }

            // Compute the target slot at the lower level's granularity
            const uint8_t new_slot = static_cast<uint8_t>(
                (nodes_[node_idx].expiry_time_ns / to_granularity) & 0xFF
            );

            // Update node metadata
            nodes_[node_idx].wheel_level = to_level;
            nodes_[node_idx].wheel_slot  = new_slot;

            // Prepend to the target slot's list — O(1)
            nodes_[node_idx].next        = wheel_[to_level][new_slot];
            wheel_[to_level][new_slot]   = node_idx;

            node_idx = next;
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // sweep_slot() — Process a single L0 slot, collecting expired entries
    //
    // Walks the slot's intrusive list. Entries whose expiry_time <= now
    // are expired and written to expired_out. Entries not yet expired
    // are kept in the slot. Cancelled entries (expiry == 0) are discarded.
    // ────────────────────────────────────────────────────────────────────
    size_t sweep_slot(uint8_t level, uint8_t slot, uint64_t current_time_ns,
                      uint32_t* expired_out, size_t max_count)
    {
        size_t count = 0;

        // Detach the list
        uint32_t node_idx       = wheel_[level][slot];
        wheel_[level][slot]     = TIMER_NULL;

        // Rebuild a list of non-expired entries to put back
        uint32_t remaining_head = TIMER_NULL;

        while (node_idx != TIMER_NULL) {
            const uint32_t next = nodes_[node_idx].next;

            if (nodes_[node_idx].expiry_time_ns == 0) {
                // Cancelled — discard silently
                node_idx = next;
                continue;
            }

            if (nodes_[node_idx].expiry_time_ns <= current_time_ns) {
                // EXPIRED — report to caller
                if (count < max_count) {
                    expired_out[count++] = nodes_[node_idx].pool_index;
                } else {
                    // Buffer full — put this node back for next sweep
                    nodes_[node_idx].next = remaining_head;
                    remaining_head = node_idx;
                }
            } else {
                // Not yet expired — keep in the slot
                nodes_[node_idx].next = remaining_head;
                remaining_head = node_idx;
            }

            node_idx = next;
        }

        // Put remaining entries back into the slot
        wheel_[level][slot] = remaining_head;
        return count;
    }

    // ────────────────────────────────────────────────────────────────────
    // flush_all_levels() — Emergency flush when watermark jumps past
    //                      the entire wheel span
    // ────────────────────────────────────────────────────────────────────
    size_t flush_all_levels(uint32_t* expired_out, size_t max_count,
                            uint64_t current_time_ns)
    {
        size_t total = 0;

        for (size_t level = 0; level < NUM_LEVELS && total < max_count; ++level) {
            for (size_t slot = 0; slot < SLOTS_PER_LEVEL && total < max_count; ++slot) {
                uint32_t node_idx       = wheel_[level][slot];
                wheel_[level][slot]     = TIMER_NULL;

                while (node_idx != TIMER_NULL && total < max_count) {
                    const uint32_t next = nodes_[node_idx].next;

                    // Skip cancelled
                    if (nodes_[node_idx].expiry_time_ns != 0 &&
                        nodes_[node_idx].expiry_time_ns <= current_time_ns)
                    {
                        expired_out[total++] = nodes_[node_idx].pool_index;
                    }

                    node_idx = next;
                }
            }
        }

        return total;
    }

    // ── Member data ──────────────────────────────────────────────────────
    size_t   node_capacity_;
    uint64_t watermark_ns_ = 0;

    // The wheel: 3 levels × 256 slots.
    // Each slot holds the head index of an intrusive linked list of TimerNodes.
    // Total overhead: 3 × 256 × 4 bytes = 3 KB (fits in L1 cache)
    std::array<std::array<uint32_t, SLOTS_PER_LEVEL>, NUM_LEVELS> wheel_;

    // Pre-allocated flat pool of timer nodes, indexed 1:1 with OrderPool
    std::vector<TimerNode> nodes_;
};
