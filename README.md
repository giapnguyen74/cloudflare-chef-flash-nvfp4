# Clef-Flash NVFP4 server

[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) quantized to NVFP4 (W4A4) and served from a Docker container with an
OpenAI-compatible API, for DGX Spark / GB10 (linux/arm64, CUDA 13). Clef-Flash is a 9B classifier: it returns a probability for every option
of every typed question in one forward pass. It does not write text.

## Run

```bash
./docker-clef.sh build          # image clef-flash-server:local, 16 GB, model baked in
HOST=127.0.0.1 ./docker-clef.sh start   # http://127.0.0.1:8100, this machine only. HOST=0.0.0.0 for every interface, or the name/address of one
                                        # interface; PORT=... for another port. There is no default HOST.
./docker-clef.sh status | stop | restart | logs   # these reuse the address the container was started on
./docker-clef.sh install-boot   # once: come back after a host reboot
```

No authentication: anyone who can reach the address can call it. Ready about 20 s after start; 10.3 GiB of GPU memory at idle (see below).

### What `build` needs and makes

A fresh clone needs [uv](https://docs.astral.sh/uv/), Docker and the CUDA 13 toolkit (`nvcc` on `PATH`) on a GB10. Nothing else from the host is used.

| Step | Result (not in git) | Notes |
|---|---|---|
| `uv sync --locked` | `.venv/` | Python 3.13 and every package, pinned by `pyproject.toml` + `uv.lock`. The first sync compiles causal-conv1d against this torch (about 6 minutes). Run it yourself to get the env for the tests. |
| `export_nvfp4.py`, when `clef-flash-nvfp4/` is missing | `clef-flash/` (19 GB), `clef-flash-nvfp4/` (9.1 GB) | Downloads the bf16 release, quantizes it, writes the weights next to the release's tokenizer, config, model code and licence, and checks that the result reloads bit-identically. Needs about 30 GB of free memory. `clef-flash/` can be deleted afterwards. |
| copy from `.venv/` | `docker/prebuilt/` (250 MB) | The compiled causal-conv1d extension; the image has no CUDA toolkit to compile it. |
| `docker build` | the image | Installs the same `uv.lock`, then adds the extension, the model and the server code. |

## API

`POST /v1/chat/completions` (OpenAI-compatible), `GET /v1/models`, `GET /health`, `GET /metrics`.

With the OpenAI API, each property of a `response_format` JSON schema is one question. The reply's `content` is a JSON object with one value
per property, and a non-standard `clef` field carries the per-option probabilities.

| Property schema | Question type | Value returned |
|---|---|---|
| `{"type": "boolean"}` | `noul` | `true` if P(true) >= 0.5 |
| `{"enum": [...]}` (optional `"x-criteria": {label: description}`) or `{"oneOf": [{"const", "description"}]}` | `choice` | the label |
| `{"type": "integer"/"number", "minimum": a, "maximum": b}` (optional `"x-levels": [description per level]`) | `score` | expected level, rounded for `integer` |

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8100/v1", api_key="unused")   # the SDK requires a value; the server ignores it
r = c.chat.completions.create(model="clef-flash",
    messages=[{"role": "user", "content": "Our checkout started returning errors and all orders are blocked!"}],
    response_format={"type": "json_schema", "json_schema": {"name": "triage", "schema": {"type": "object", "properties": {
        "department": {"description": "Which team should handle the message?", "enum": ["billing", "technical", "sales"]},
        "outage":     {"type": "boolean", "description": "Is a service down?"},
        "urgency":    {"type": "integer", "minimum": 1, "maximum": 5, "description": "How urgent is this?"}}}}})
print(r.choices[0].message.content)      # {"department": "technical", "outage": true, "urgency": 5}
print(r.model_extra["clef"]["answers"])  # per-option probabilities
```

Also supported: `tools` (the chosen function's `parameters` are filled the same way and returned as a tool call; with several tools, name one
in `tool_choice`), `stream: true` (the answer arrives as one chunk), images as `data:` URLs and multi-turn `messages`. Plain chat with no
schema, free-text string properties and remote image URLs get HTTP 400 with an explanation.

The model's own question format is accepted as an extra body field, `questions`, in place of a schema. It allows a description per option
and returns P(true) for a `noul` question instead of a boolean:

```bash
curl -s 127.0.0.1:8100/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "Our checkout started returning errors and orders are blocked."}],
  "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?",
                               "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
                "outage": {"type": "noul", "instructions": "Is a service down?"}}}'
```

## Measured (2026-10-06, GB10, in the container)

| Input tokens | Latency (median) | Input tok/s |
|---|---|---|
| 158 | 42 ms | 3,800 |
| 858 | 152 ms | 5,650 |
| 3,658 | 486 ms | 7,530 |
| 7,158 | 951 ms | 7,530 |

Short requests: 24 / 37 / 43 req/s at 1 / 8 / 32 clients. The container and a host-run server differ by less than 2.5%.
Accuracy on 300 AG News + 300 Banking77 test samples: 91.0% / 94.3% (unquantized bf16: 92.3% / 94.0%). Small, easy tests: they show the
quantized model tracks the original here, not how accurate it is on your data.

### GPU memory

Memory of the server process as `nvidia-smi` reports it, from a fresh start, running the workloads in this order. The process keeps what it
has reached (PyTorch caches freed blocks), so each row is the high-water mark so far, not a per-request cost.

| Workload | GPU memory |
|---|---|
| Idle after start and warm-up | 10.3 GiB |
| Requests up to 3,658 input tokens, one at a time; 158-token requests at 1 / 8 / 32 clients | 10.3 GiB |
| 7,158 input tokens, one at a time | 11.1 GiB |
| 3,658 or 7,158 input tokens, 8 clients | 11.6 GiB |
| 16,384 input tokens (the maximum; longer inputs are cut to it), 1 or 4 clients | 13.8 GiB |

Of the idle figure, 8.6 GiB (9.2 GB) is the model as PyTorch counts it; the rest is the CUDA context and allocator cache. PyTorch's own
allocations are capped at `CLEF_GPU_FRACTION` of the GPU's memory (default 0.14, 17 GiB here); a request that would exceed it gets HTTP 503.
None of these workloads reached the cap. On top of the GPU memory, `docker stats` showed 4.5 GiB of RAM for the container.

## Things to know

- **Inputs are text, or base64 for images. The server never follows a link.** A URL inside a message, a schema or a question is plain text
  to the model. An image must be base64 in a `data:` URL; a link in its place gets HTTP 400. No other request field reaches the model code.
- **Static activation scales.** Quantization scales were calibrated on news and banking samples plus synthetic requests, with 4x headroom
  (`clef_fast_calib.json`). Inputs far outside that saturate and lose accuracy (forcing it dropped Banking77 to 88.7%).
  `CLEF_ACT_SCALE=dynamic ./docker-clef.sh restart` removes the assumption for about 10% less speed.
- **Answers depend on which questions are asked together.** "Is a service down?" for the message above scores 0.10 alone and `true` next to
  `department` and `urgency`; the unquantized model does the same. Keep the question set fixed when you compare outputs or set thresholds.
- **Pinned to one platform.** aarch64, Python 3.13, torch 2.14.1+cu130, torchao 0.18.0 (`pyproject.toml`, `uv.lock`). The model file is not a
  transformers checkpoint; it loads only through `clef_fast.load_packed()`.
- **GPU access is CDI-only on DGX OS** (`--device nvidia.com/gpu=all`), which the script already uses.

## Files

| Path | Purpose |
|---|---|
| `clef-flash-nvfp4/` | Not in git, made by the export: the quantized weights plus tokenizer, processor, head config, model code and licence |
| `server.py`, `clef_fast.py`, `openai_compat.py`, `clef_fast_calib.json` | The server, the fused NVFP4 backend, the OpenAI mapping, calibration |
| `pyproject.toml`, `uv.lock` | The Python environment, for the host's `.venv/` and for the image |
| `docker/`, `docker-clef.sh`, `.dockerignore` | Dockerfile, build/run/boot script |
| `export_nvfp4.py` | One-time export of the quantized weights from the original release |
| `test_openai.py`, `test_server.py`, `bench_http.py` | 17 checks through the `openai` SDK, load test, latency benchmark. Run with `uv run python test_openai.py`; `CLEF_URL=http://host:port` for a server that is not on `127.0.0.1:8100` |

## Licence

The model is Apache-2.0 (`clef-flash-nvfp4/LICENSE`); `clef-flash-nvfp4/NVFP4_README.md` states what was changed.
