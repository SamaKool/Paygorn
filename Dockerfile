# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build — Memory-Safe C++ Strategy
#
# STAGE 1 (builder): Install Python deps + attempt to use pre-compiled .so.
#   • If a pre-compiled hft_auditor*.so exists in the build context it is used
#     directly — no CMake, no RAM spike.
#   • If the .so is missing, CMake compiles with -j1 and -O1 to stay under the
#     8 GB RAM limit on HuggingFace Spaces (peak ~1.2 GB vs ~5 GB for -O3).
#
# STAGE 2 (runtime): Minimal image — only the venv, app code, and the .so.
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# git is needed for VCS deps; cmake + build-essential needed only for fallback
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=in-repo
ARG ENV_NAME=fin_auditor

# Copy environment code into the builder context
COPY . /app/env
WORKDIR /app/env

# Ensure uv is available
RUN pip install uv

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
    uv sync --frozen --no-install-project --no-editable; \
    else \
    uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
    uv sync --frozen --no-editable; \
    else \
    uv sync --no-editable; \
    fi

# ── C++ Engine: pre-built .so first, memory-safe CMake fallback ──────────────
#
# Decision logic (runs at Docker build time, not container start):
#   1. Check if a compiled hft_auditor*.so already lives in the build context.
#   2. If YES → emit a success message and skip CMake entirely.
#   3. If NO  → install build tools and compile with:
#        -j1           (single compile job — caps peak RAM at ~1.5 GB)
#        -O1           (lower optimisation — cuts peak RAM from ~5 GB to ~1.2 GB)
#        DOCKER_SAFE_BUILD=ON (CMakeLists.txt option that overrides -O3 with -O1)
#
RUN set -e; \
    SO_FILE=$(ls /app/env/hft_auditor*.so 2>/dev/null | head -1); \
    if [ -n "$SO_FILE" ]; then \
        echo "[BUILD] ✓ Pre-compiled .so found at $SO_FILE — skipping CMake."; \
    else \
        echo "[BUILD] No pre-compiled .so found. Falling back to memory-safe CMake build..."; \
        apt-get update && \
        apt-get install -y --no-install-recommends cmake build-essential && \
        rm -rf /var/lib/apt/lists/*; \
        cd /app/env && \
        python build_engine.py --docker-safe; \
    fi

# ── Final runtime stage ───────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the Python virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy the full environment (source + data + .so binary)
COPY --from=builder /app/env /app/env

# Activate the venv
ENV PATH="/app/.venv/bin:$PATH"

# Make all modules importable (models, server, hft_auditor .so, inference.py)
ENV PYTHONPATH="/app/env:$PYTHONPATH"


# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]