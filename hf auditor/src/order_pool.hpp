#pragma once
// ============================================================================
// OrderPool — O(1) Flat-Array Trade Storage
// ============================================================================
//
// ARCHITECTURE:
//   Dense, contiguous trade IDs are mapped to slots via bitwise AND:
//       index = trade_id & (capacity - 1)
//   This gives single-cycle O(1) insert and lookup with zero branching.
//
// MEMORY LAYOUT (Structure-of-Arrays):
//   Trade data:  std::vector<TradeSlot>   — 32 bytes/slot, aligned for AVX loads
//   Trade state: std::vector<SlotState>   — 1 byte/slot, packed dense
//
//   SoA is chosen over AoS because:
//   1. Scanning states (finding all ACTIVE trades for the observation matrix)
//      touches only the state array: 64 states per cache line vs. 1 per 32-byte slot.
//   2. AVX2 can process 32 state bytes in a single VPCMPEQB instruction.
//
// ZERO-ALLOCATION GUARANTEE:
//   Both vectors are resize()'d at construction. No runtime heap calls.
//
// ============================================================================

#include <vector>
#include <cstdint>
#include <cassert>

// ────────────────────────────────────────────────────────────────────────────
// Slot States — kept as uint8_t enum for dense packing
// ────────────────────────────────────────────────────────────────────────────
enum class SlotState : uint8_t {
    EMPTY       = 0,   // Slot has never been written, or was evicted
    ACTIVE      = 1,   // Trade is live, awaiting reconciliation
    RECONCILED  = 2,   // Trade matched a bank receipt — success
    EXPIRED     = 3    // Trade exceeded Δ_max — anomaly sent to RL agent
};

// ────────────────────────────────────────────────────────────────────────────
// TradeSlot — 32-byte cache-aligned POD struct
// ────────────────────────────────────────────────────────────────────────────
//
// Layout (total = 32 bytes, zero padding waste):
//   [0..7]   trade_id         uint64_t    Unique contiguous ID from ingestion gateway
//   [8..15]  price            int64_t     Fixed-point price (cents/pips/satoshis)
//   [16..19] quantity         int32_t     Signed quantity (negative = sell)
//   [20..23] counterparty_id  uint32_t    Counterparty identifier for risk scoring
//   [24..31] timestamp_ns     uint64_t    Event-time in nanoseconds since epoch
//
struct alignas(32) TradeSlot {
    uint64_t trade_id;          // 8 bytes
    int64_t  price;             // 8 bytes
    int32_t  quantity;          // 4 bytes
    uint32_t counterparty_id;   // 4 bytes
    uint64_t timestamp_ns;      // 8 bytes
    // ─────────────────────── = 32 bytes total, no padding needed
};
static_assert(sizeof(TradeSlot) == 32, "TradeSlot must be exactly 32 bytes for cache alignment");

// ────────────────────────────────────────────────────────────────────────────
// OrderPool — O(1) flat-array trade storage
// ────────────────────────────────────────────────────────────────────────────
class OrderPool {
public:
    // ────────────────────────────────────────────────────────────────────────
    // Constructor: pre-allocates the entire pool at startup.
    // capacity MUST be a power of 2 so that (trade_id & mask_) is a
    // single-cycle bitwise AND instead of an expensive integer modulo.
    // ────────────────────────────────────────────────────────────────────────
    explicit OrderPool(size_t capacity)
        : capacity_(capacity)
        , mask_(capacity - 1)
    {
        // Validate power-of-2 invariant
        assert((capacity & (capacity - 1)) == 0 &&
               "OrderPool capacity must be a power of 2");

        // ZERO-ALLOCATION GUARANTEE: all memory is grabbed here at startup.
        // resize() default-constructs all elements in-place.
        slots_.resize(capacity);
        states_.resize(capacity, SlotState::EMPTY);
    }

    // ────────────────────────────────────────────────────────────────────────
    // insert() — O(1) direct-index insertion
    //
    // For dense contiguous IDs, (trade_id & mask_) gives the unique slot.
    // If two trade_ids map to the same slot (only possible if IDs span
    // more than capacity), the newer trade overwrites the older one.
    // This is acceptable: the old trade would have expired via the timer
    // wheel before its slot index is reused.
    // ────────────────────────────────────────────────────────────────────────
    bool insert(uint64_t trade_id, int64_t price, int32_t quantity,
                uint32_t counterparty_id, uint64_t timestamp_ns)
    {
        const size_t idx = trade_id & mask_;

        // Direct write — no branching, no probing
        auto& slot    = slots_[idx];
        slot.trade_id        = trade_id;
        slot.price           = price;
        slot.quantity        = quantity;
        slot.counterparty_id = counterparty_id;
        slot.timestamp_ns    = timestamp_ns;

        // Only increment active count if slot wasn't already ACTIVE
        // (handles the overwrite case)
        if (states_[idx] != SlotState::ACTIVE) {
            ++active_count_;
        }
        states_[idx] = SlotState::ACTIVE;

        return true;
    }

    // ────────────────────────────────────────────────────────────────────────
    // lookup() — O(1) direct-index lookup
    //
    // Returns a pointer to the slot if found and ACTIVE, nullptr otherwise.
    // The pointer is valid until the slot is overwritten (no invalidation
    // during normal operation since the pool is stable).
    // ────────────────────────────────────────────────────────────────────────
    TradeSlot* lookup(uint64_t trade_id) {
        const size_t idx = trade_id & mask_;

        // Guard: verify the slot actually holds this trade_id
        // (necessary because another ID could hash to the same slot)
        if (states_[idx] == SlotState::ACTIVE &&
            slots_[idx].trade_id == trade_id)
        {
            return &slots_[idx];
        }
        return nullptr;
    }

    const TradeSlot* lookup(uint64_t trade_id) const {
        const size_t idx = trade_id & mask_;
        if (states_[idx] == SlotState::ACTIVE &&
            slots_[idx].trade_id == trade_id)
        {
            return &slots_[idx];
        }
        return nullptr;
    }

    // ────────────────────────────────────────────────────────────────────────
    // State accessors
    // ────────────────────────────────────────────────────────────────────────
    SlotState get_state(size_t idx) const { return states_[idx]; }

    void set_state(size_t idx, SlotState new_state) {
        // Maintain the active_count_ invariant
        if (states_[idx] == SlotState::ACTIVE && new_state != SlotState::ACTIVE) {
            --active_count_;
        } else if (states_[idx] != SlotState::ACTIVE && new_state == SlotState::ACTIVE) {
            ++active_count_;
        }
        states_[idx] = new_state;
    }

    // ────────────────────────────────────────────────────────────────────────
    // Direct slot access (for timer wheel back-pointers and observation scan)
    // ────────────────────────────────────────────────────────────────────────
    const TradeSlot& slot_at(size_t idx) const { return slots_[idx]; }
    TradeSlot&       slot_at(size_t idx)       { return slots_[idx]; }

    // ────────────────────────────────────────────────────────────────────────
    // Capacity and statistics
    // ────────────────────────────────────────────────────────────────────────
    size_t capacity()     const { return capacity_; }
    size_t mask()         const { return mask_; }
    size_t active_count() const { return active_count_; }

    // Raw data pointer for potential zero-copy exposure
    TradeSlot*       data()       { return slots_.data(); }
    const TradeSlot* data() const { return slots_.data(); }

private:
    size_t capacity_;                    // Always power of 2
    size_t mask_;                        // capacity_ - 1, for bitwise AND
    size_t active_count_ = 0;            // Number of ACTIVE slots

    std::vector<TradeSlot> slots_;       // 32 bytes × capacity — trade data
    std::vector<SlotState> states_;      // 1 byte  × capacity — slot states
};
