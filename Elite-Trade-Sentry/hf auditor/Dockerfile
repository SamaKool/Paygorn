# ── Stage 1: Build C++ Backend ───────────────────────────────────────
# We use Python 3.11-slim as a base to ensure the correct Python development
# headers are present for Nanobind.
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies for C++20 and CMake
# We need ninja-build for modern, fast parallel compilation.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install build-time requirements for Nanobind
RUN pip install --no-cache-dir nanobind numpy

# Copy build configuration and source code
COPY CMakeLists.txt .
COPY src/ src/

# Build the C++ extension module
# Note: we disable Benchmarks in Docker to minimize the build image size.
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBENCHMARK_ENABLE_TESTING=OFF \
    -DBENCHMARK_ENABLE_WERROR=OFF \
    && cmake --build build --config Release

# ── Stage 2: Runtime Environment ──────────────────────────────────────
# Switch to a clean, minimal runtime image to keep docker size small.
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies for FastAPI and NumPy
RUN pip install --no-cache-dir fastapi uvicorn numpy nanobind

# Copy the built C++ library from the builder stage
# Extension modules are .so files on Linux.
COPY --from=builder /app/build/hft_auditor*.so .

# Copy the FastAPI server source
COPY api.py .

# Expose port for FastAPI
EXPOSE 8000

# Launch the uvicorn FastAPI server
# We use 0.0.0.0 to make it reachable from outside the container
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
