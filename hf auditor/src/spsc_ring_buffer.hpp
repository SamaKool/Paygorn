#pragma once
// ============================================================================
// SPSCRingBuffer — Lock-Free Single-Producer Single-Consumer Queue
// ============================================================================
//
// ARCHITECTURE:
//   A bounded, lock-free ring buffer for passing trades from the ingestion
//   thread (producer) to the engine thread (consumer). This is the only
//   cross-thread data path in the entire system.
//
// MEMORY ORDERING:
//   Uses std::memory_order_acquire/release — the minimum fence strength
//   that guarantees correctness for SPSC. Sequential consistency
//   (memory_order_seq_cst) is avoided because it inserts unnecessary
//   MFENCE instructions on x86 that cost ~30ns each.
//
//   Producer: writes data, then head_.store(release)
//             → ensures all writes to buffer_[idx] are visible before
//               the consumer sees the incremented head.
//
//   Consumer: reads head_.load(acquire)
//             → ensures it sees all writes the producer made before
//               the matching head_.store(release).
//
// FALSE-SHARING PREVENTION:
//   head_ and tail_ are placed on separate 128-byte aligned regions.
//   Why 128 and not 64? Intel CPUs prefetch cache lines in pairs
//   (spatial prefetcher). Separating by 128 bytes ensures head and tail
//   never share a prefetch pair, eliminating false sharing even on
//   aggressive prefetch hardware.
//
// CAPACITY:
//   Must be a power of 2 so that index wrapping uses bitwise AND:
//       real_index = sequence_number & (Capacity - 1)
//   This compiles to a single AND instruction instead of an integer
//   division (which costs 20-90 cycles on modern x86).
//
// ZERO-ALLOCATION GUARANTEE:
//   The internal buffer is allocated once at construction via
//   std::vector::resize(). No heap calls during push/pop.
//
// ============================================================================

#include <atomic>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <type_traits>

// ────────────────────────────────────────────────────────────────────────────
// RingEntry — 32-byte aligned ingestion record
// ────────────────────────────────────────────────────────────────────────────
//
// This struct matches the data produced by the ingestion gateway.
// It's intentionally identical in layout to OrderPool::TradeSlot so
// the consumer can memcpy directly (or the compiler can optimize
// the field-by-field copy into a single AVX store).
//
// Layout (total = 32 bytes):
//   [0..7]   trade_id         uint64_t
//   [8..15]  price            int64_t
//   [16..19] quantity         int32_t
//   [20..23] counterparty_id  uint32_t
//   [24..31] timestamp_ns     uint64_t
//
struct alignas(32) RingEntry {
    uint64_t trade_id;          // 8 bytes
    int64_t  price;             // 8 bytes
    int32_t  quantity;          // 4 bytes
    uint32_t counterparty_id;   // 4 bytes
    uint64_t timestamp_ns;      // 8 bytes
    // ────────────────────── = 32 bytes total
};
static_assert(sizeof(RingEntry) == 32, "RingEntry must be exactly 32 bytes");

// ────────────────────────────────────────────────────────────────────────────
// SPSCRingBuffer<Capacity> — Lock-free SPSC queue
// ────────────────────────────────────────────────────────────────────────────
template <size_t Capacity>
class SPSCRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "SPSCRingBuffer capacity must be a power of 2");
    static_assert(Capacity > 0, "Capacity must be > 0");

    // Compile-time mask for index wrapping: Capacity - 1
    static constexpr size_t MASK = Capacity - 1;

public:
    // ────────────────────────────────────────────────────────────────────
    // Constructor: pre-allocate the ring buffer
    //
    // The buffer is heap-allocated via std::vector at construction time.
    // This is the ONLY allocation. All subsequent push/pop operations
    // are zero-allocation.
    // ────────────────────────────────────────────────────────────────────
    SPSCRingBuffer() : buffer_(Capacity) {}

    // Non-copyable, non-movable (atomics can't be copied)
    SPSCRingBuffer(const SPSCRingBuffer&) = delete;
    SPSCRingBuffer& operator=(const SPSCRingBuffer&) = delete;
    SPSCRingBuffer(SPSCRingBuffer&&) = delete;
    SPSCRingBuffer& operator=(SPSCRingBuffer&&) = delete;

    // ────────────────────────────────────────────────────────────────────
    // try_push() — Producer API
    //
    // Called from the ingestion thread. Attempts to enqueue one trade.
    // Returns false if the buffer is full (back-pressure signal to the
    // producer that the consumer isn't keeping up).
    //
    // Memory ordering:
    //   1. Load tail_ with ACQUIRE to see the consumer's latest progress
    //   2. Write the entry to buffer_[head & MASK]
    //   3. Store head_+1 with RELEASE to publish the new entry
    // ────────────────────────────────────────────────────────────────────
    bool try_push(uint64_t trade_id, int64_t price, int32_t quantity,
                  uint32_t counterparty_id, uint64_t timestamp_ns)
    {
        // Read our own head (relaxed — we're the only writer)
        const uint64_t head = head_.load(std::memory_order_relaxed);

        // Read the consumer's tail (acquire — synchronize with consumer's
        // release store to see freed slots)
        const uint64_t tail = tail_.load(std::memory_order_acquire);

        // Check if buffer is full
        if (head - tail >= Capacity) {
            return false;  // Back-pressure: consumer must drain
        }

        // Write the entry — this is safe because no other thread writes
        // to this slot (the consumer has already consumed past it)
        auto& entry         = buffer_[head & MASK];
        entry.trade_id      = trade_id;
        entry.price         = price;
        entry.quantity       = quantity;
        entry.counterparty_id = counterparty_id;
        entry.timestamp_ns  = timestamp_ns;

        // Publish: make the entry visible to the consumer.
        // The RELEASE fence ensures all writes above are committed
        // before the consumer sees the incremented head.
        head_.store(head + 1, std::memory_order_release);

        return true;
    }

    // ────────────────────────────────────────────────────────────────────
    // drain() — Consumer API (batch)
    //
    // Called from the engine thread. Drains ALL available entries in a
    // tight loop, invoking the callback for each entry. This amortizes
    // the atomic load of head_ across the entire batch.
    //
    // The callback signature must be: void(const RingEntry&)
    //
    // Memory ordering:
    //   1. Load head_ with ACQUIRE to see the producer's latest progress
    //   2. Read entries from buffer_[tail..head)
    //   3. Store new tail with RELEASE to free slots for the producer
    //
    // Returns: number of entries drained
    // ────────────────────────────────────────────────────────────────────
    template <typename Callback>
    size_t drain(Callback&& cb) {
        // Read our own tail (relaxed — we're the only writer of tail_)
        const uint64_t tail = tail_.load(std::memory_order_relaxed);

        // Read the producer's head (acquire — synchronize with producer's
        // release store to see published entries)
        const uint64_t head = head_.load(std::memory_order_acquire);

        // Nothing to drain?
        if (tail == head) {
            return 0;
        }

        // Drain all available entries in a tight loop
        size_t count = 0;
        for (uint64_t i = tail; i < head; ++i) {
            const auto& entry = buffer_[i & MASK];
            cb(entry);
            ++count;
        }

        // Publish: free all drained slots for the producer.
        // The RELEASE fence ensures the callback has fully processed
        // each entry before the producer can overwrite the slot.
        tail_.store(head, std::memory_order_release);

        return count;
    }

    // ────────────────────────────────────────────────────────────────────
    // try_pop() — Consumer API (single)
    //
    // Pops a single entry. Useful for low-latency processing where
    // batch drain adds too much latency to the first entry.
    // ────────────────────────────────────────────────────────────────────
    bool try_pop(RingEntry& out) {
        const uint64_t tail = tail_.load(std::memory_order_relaxed);
        const uint64_t head = head_.load(std::memory_order_acquire);

        if (tail == head) {
            return false;  // Empty
        }

        out = buffer_[tail & MASK];
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    // ────────────────────────────────────────────────────────────────────
    // Status queries (approximate — inherently racy across threads,
    // but useful for monitoring and back-pressure decisions)
    // ────────────────────────────────────────────────────────────────────
    size_t size() const {
        const uint64_t head = head_.load(std::memory_order_acquire);
        const uint64_t tail = tail_.load(std::memory_order_acquire);
        return static_cast<size_t>(head - tail);
    }

    bool empty() const { return size() == 0; }

    bool full() const { return size() >= Capacity; }

    static constexpr size_t capacity() { return Capacity; }

private:
    // ── CRITICAL: False-sharing prevention ──────────────────────────────
    //
    // head_ is written ONLY by the producer thread.
    // tail_ is written ONLY by the consumer thread.
    //
    // alignas(128) places them on separate 128-byte regions, ensuring
    // they never share a cache line pair. Without this, every producer
    // write to head_ would invalidate the consumer's cache line
    // containing tail_, and vice versa — a catastrophic performance bug
    // called "false sharing" that can reduce throughput by 10-50×.
    //
    // Why 128 instead of 64 (one cache line)?
    // Intel's spatial prefetcher fetches adjacent cache lines in pairs.
    // A 64-byte separation still allows the prefetcher to pull both
    // lines into the same core's L1, causing false sharing.
    // 128 bytes guarantees isolation.
    //
    alignas(128) std::atomic<uint64_t> head_{0};   // Producer writes
    alignas(128) std::atomic<uint64_t> tail_{0};   // Consumer writes

    // The ring buffer storage. Heap-allocated once at construction.
    // Each entry is 32-byte aligned for potential AVX loads.
    std::vector<RingEntry> buffer_;
};
