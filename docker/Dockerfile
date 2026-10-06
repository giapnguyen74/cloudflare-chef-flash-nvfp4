# Clef-Flash server (NVFP4 fast backend) for DGX Spark / GB10: linux/arm64, CUDA 13 wheels, the same uv.lock as the host's .venv/.
# Build with ./docker-clef.sh build from the repository root (it stages the prebuilt causal-conv1d extension). The pre-quantized NVFP4 model (clef-flash-nvfp4/, 9 GB,
# made by export_nvfp4.py) is baked in, so the container needs no mounts. Model licence: Apache-2.0, /app/clef-flash-nvfp4/LICENSE.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
# gcc/g++: torch.compile (inductor + triton) builds small launcher stubs at first use
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gcc g++ libc6-dev && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH
WORKDIR /app
COPY pyproject.toml uv.lock /app/
# the locked packages, minus the test tools and causal-conv1d. The cache mount keeps the downloads (several GB) out of the image and across rebuilds.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-package causal-conv1d
# causal-conv1d needs the CUDA toolkit to compile; reuse the extension built in the host's .venv/ against this exact torch (staged by docker-clef.sh)
COPY docker/prebuilt/ /opt/venv/lib/python3.13/site-packages/
# weights first: this 9 GB layer is reused when only the server code changes
COPY clef-flash-nvfp4/ /app/clef-flash-nvfp4/
COPY server.py clef_fast.py openai_compat.py clef_fast_calib.json /app/
ENV TORCHINDUCTOR_CACHE_DIR=/cache/inductor TRITON_CACHE_DIR=/cache/triton HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
VOLUME /cache
EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s CMD curl -sf http://127.0.0.1:8100/health || exit 1
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8100"]
