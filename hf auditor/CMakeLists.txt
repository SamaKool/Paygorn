cmake_minimum_required(VERSION 3.15)
project(HighFrequencyAuditor CXX)

# ── C++20 Standard ──────────────────────────────────────────────────────────
# Required for:
#   - std::atomic with memory_order_acquire/release (C++11, but C++20 for consistency)
#   - constexpr if, structured bindings, fold expressions
#   - alignas(128) on class members
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Platform-Specific Optimization Flags ────────────────────────────────────
#
# The original CMakeLists used GCC/Clang flags (-O3, -march=native) which
# fail silently on MSVC. We now detect the compiler and set appropriate flags.
#
if(MSVC)
    # /O2          — Full optimization (equivalent to GCC -O2)
    # /arch:AVX2   — Enable AVX2 SIMD instructions for 256-bit vectorization
    # /DNDEBUG     — Disable assert() in release builds
    # /EHsc        — Standard C++ exception handling (required for nanobind)
    # /GL          — Whole-program optimization (link-time code generation)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /arch:AVX2 /DNDEBUG /EHsc /GL")
else()
    # GCC/Clang flags
    # -O3           — Aggressive optimization including vectorization
    # -march=native — Use all CPU-specific instructions available
    # -DNDEBUG      — Disable assert()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
endif()

# ── Thread Support ──────────────────────────────────────────────────────────
# Required for:
#   - std::atomic (the SPSC ring buffer's acquire/release semantics)
#   - std::thread (if the producer thread is managed from C++)
#   - On Linux/macOS, links -lpthread. On Windows, threading is built-in.
find_package(Threads REQUIRED)

# ── Python + Nanobind ───────────────────────────────────────────────────────
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(nanobind)

# ── Build the Python Extension Module ───────────────────────────────────────
#
# nanobind_add_module() creates a .pyd (Windows) or .so (Linux) Python
# extension. The module is named "hft_auditor" and Python can import it as:
#     import hft_auditor
#
# All engine code is in header files included by auditor.cpp:
#   - src/order_pool.hpp
#   - src/timer_wheel.hpp
#   - src/spsc_ring_buffer.hpp
#   - src/reconciliation_engine.hpp
#
nanobind_add_module(hft_auditor src/auditor.cpp)

# Link thread library (required for atomics on some platforms)
target_link_libraries(hft_auditor PRIVATE Threads::Threads)