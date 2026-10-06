"""Clef-Flash server: an OpenAI-compatible /v1/chat/completions (see openai_compat.py), with warm-up,
micro-batching and bounded GPU memory. Backends: NVFP4 (default), NF4, bf16.

Endpoints:  POST /v1/chat/completions   GET /v1/models   GET /health   GET /metrics

Env:
  CLEF_MODEL_NAME      model id reported by /v1/models (default clef-flash)
  CLEF_QUANT           nvfp4 (default; torchao W4A4 + fused kernels, see clef_fast.py) | nf4 (bitsandbytes) | bf16
  CLEF_PACKED_DIR      pre-quantized NVFP4 model directory (default clef-flash-nvfp4). If it is missing, nvfp4 quantizes clef-flash/ at startup.
  CLEF_ACT_SCALE       nvfp4 only: static (default; calibrated activation scales from clef_fast_calib.json, x4 headroom) | dynamic (about 10% slower,
                       no calibration assumption)
  CLEF_GPU_FRACTION    cap on this process's GPU memory as a fraction of total (default 0.14)
  CLEF_MAX_BATCH       max requests per forward (default 8)
  CLEF_MAX_PAD_TOKENS  max padded tokens per forward (default 8192); one longer request still runs alone
  CLEF_MAX_STATE_TOKENS  optional cap on state tokens per request
"""
import asyncio, base64, io, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from PIL import Image

QUANT = os.environ.get("CLEF_QUANT", "nvfp4")
PACKED = os.environ.get("CLEF_PACKED_DIR", "clef-flash-nvfp4")     # pre-quantized model (export_nvfp4.py); used by the nvfp4 backend when present
USE_PACKED = QUANT == "nvfp4" and Path(PACKED, "model_nvfp4.safetensors").exists()
P = PACKED if USE_PACKED else "clef-flash"; sys.path.insert(0, P)
import joint_schema_model as J
from joint_schema_model import QUESTION_TYPES, collate_records, encode_record, systemone_answer

ACT_SCALE = os.environ.get("CLEF_ACT_SCALE", "static")
GPU_FRACTION = float(os.environ.get("CLEF_GPU_FRACTION", "0.14"))
MAX_BATCH = int(os.environ.get("CLEF_MAX_BATCH", "8"))
MAX_PAD_TOKENS = int(os.environ.get("CLEF_MAX_PAD_TOKENS", "8192"))
MAX_STATE_TOKENS = int(os.environ["CLEF_MAX_STATE_TOKENS"]) if os.environ.get("CLEF_MAX_STATE_TOKENS") else None
MAX_LEN = 16384
MODEL_NAME = os.environ.get("CLEF_MODEL_NAME", "clef-flash")
STATS = {"requests": 0, "errors": 0, "batches": 0, "batched_requests": 0, "input_tokens": 0, "latency_s": 0.0, "gpu_s": 0.0}

app = FastAPI()
model = processor = queue = worker_task = None
encode_pool = ThreadPoolExecutor(2)   # tokenization / image preprocessing, off the event loop
gpu_pool = ThreadPoolExecutor(1)      # exactly one forward pass at a time


def load():
    if QUANT == "bf16":
        return J.load_release_model(P, device="cuda")
    if QUANT == "nvfp4":
        import clef_fast as CF
        m, proc = CF.load_packed(P) if USE_PACKED else CF.load_nvfp4(P)   # packed: seconds, no bf16 checkpoint needed
        CF.apply_fast(m)
        calib = Path(__file__).with_name("clef_fast_calib.json")
        if ACT_SCALE == "static" and calib.exists():
            CF.Act.HEADROOM = 4.0; CF.set_static(m, json.loads(calib.read_text()))
        return m, proc
    from safetensors.torch import load_file
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           llm_int8_skip_modules=["visual", "lm_head"])
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(P, dtype=torch.bfloat16, device_map={"": "cuda"}, quantization_config=q)
    backbone.config.use_cache = False
    head = J.JointSchemaHead(**json.loads(Path(P, "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(Path(P, "joint_head.safetensors")), strict=True)
    return J.ClefModel(backbone, head.to(device="cuda", dtype=torch.bfloat16)).eval(), AutoProcessor.from_pretrained(P)


def validate(req):
    qs = req.get("questions")
    if "state" not in req or not isinstance(qs, dict) or not qs:
        raise ValueError("state and at least one question are required")
    for qid, q in qs.items():
        if q.get("type") not in QUESTION_TYPES:
            raise ValueError(f"{qid}: type must be noul, choice, or score")
        if q["type"] != "noul" and not q.get("criteria"):
            raise ValueError(f"{qid}: criteria must not be empty")


def prepare(req):
    """Validate, decode images and tokenize. CPU only; runs in encode_pool."""
    validate(req)
    # Only these fields reach the model code. Images are decoded here from base64; a string that got through to the processor (as "videos",
    # or via "media_kwargs") would be loaded by transformers as a URL or a local file path.
    if "videos" in req or "media_kwargs" in req:
        raise ValueError("videos and media_kwargs are not supported; send images as base64")
    r = {k: req[k] for k in ("model", "state", "questions", "id") if k in req}
    if req.get("images"):
        if not isinstance(req["images"], list) or not all(isinstance(s, str) for s in req["images"]):
            raise ValueError("images must be a list of base64 strings")
        r["images"] = [Image.open(io.BytesIO(base64.b64decode(s.split(",")[-1]))).convert("RGB") for s in req["images"]]
    kw = {"max_state_tokens": MAX_STATE_TOKENS} if MAX_STATE_TOKENS else {}
    return r, encode_record(processor.tokenizer, r, max_length=MAX_LEN, processor=processor, **kw)


@torch.inference_mode()
def run_batch(items):
    """items: [(request, EncodedRecord)] -> list of response dicts."""
    dev = next(model.parameters()).device
    logits = model(collate_records([e for _, e in items], processor.tokenizer.pad_token_id, dev))
    if QUANT == "nvfp4":                                      # drop the quantization cache's reference to this batch's activations
        import clef_fast as CF
        CF.Act.last = (None, None); CF._views.clear()
    out = []
    for (req, enc), ql in zip(items, logits):
        answers = {q.question_id: systemone_answer(req["questions"][q.question_id],
                   dict(zip(q.option_ids, l.float().softmax(-1).tolist()))) for q, l in zip(enc.questions, ql)}
        out.append({"model": req.get("model", "clef-flash"), "answers": answers,
                    "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0}})
    return out


def split(batch):
    """Sorted-by-length batch -> chunks whose padded size stays under MAX_PAD_TOKENS."""
    while batch:
        n = 1
        while n < len(batch) and n < MAX_BATCH and (n + 1) * len(batch[n][1].input_ids) <= MAX_PAD_TOKENS:
            n += 1
        yield batch[:n]
        batch = batch[n:]


def settle(fut, result=None, error=None):
    """Resolve a request's future unless its caller is already gone: setting a cancelled future raises InvalidStateError, which would end the worker."""
    if fut.done(): return
    if error is not None: fut.set_exception(error)
    else: fut.set_result(result)


async def worker():
    loop = asyncio.get_running_loop()
    while True:
        # No fixed wait: when idle a request goes straight to the GPU; while a forward pass runs,
        # new requests pile up in the queue and the next iteration drains them as one batch.
        batch = [await queue.get()]
        while not queue.empty():
            batch.append(queue.get_nowait())
        batch.sort(key=lambda b: len(b[1].input_ids))
        for chunk in split(batch):
            try:
                t = time.perf_counter(); res = await loop.run_in_executor(gpu_pool, run_batch, [(r, e) for r, e, _ in chunk])
                STATS["batches"] += 1; STATS["batched_requests"] += len(chunk); STATS["gpu_s"] += time.perf_counter() - t
                for (_, _, fut), r in zip(chunk, res): settle(fut, r)
            except torch.cuda.OutOfMemoryError as ex:
                torch.cuda.empty_cache()
                for _, _, fut in chunk: settle(fut, error=HTTPException(503, "GPU memory limit reached; retry or send a shorter request"))
            except Exception as ex:
                for _, _, fut in chunk: settle(fut, error=ex)


@app.on_event("startup")
async def startup():
    global model, processor, queue, worker_task
    model, processor = load()                                 # nvfp4 loads bf16 (19 GB) and quantizes on the GPU, so the cap goes on afterwards
    import gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
    img = Image.new("RGB", (224, 224), "white")
    # warm-up: kernel autotuning and the allocator, for short/long text, multi-question and image inputs
    for state, n, im in [("hi", 1, None), ("filler text " * 300, 3, None), ("filler text " * 3000, 1, None), ("receipt", 1, img)]:
        qs = {f"q{i}": {"type": "noul", "instructions": f"Is {i} true?"} for i in range(n)}
        r = {"state": state, "questions": qs, **({"images": [im]} if im else {})}
        run_batch([(r, encode_record(processor.tokenizer, r, processor=processor))])
    # batched shapes too: the first multi-request batch otherwise pays about a second of kernel compilation
    for bs in (2, 8):
        rs = [{"state": "short message " * (i + 1), "questions": {"q": {"type": "choice", "instructions": "Which?", "criteria": {"a": "first", "b": "second", "c": "third"}}}} for i in range(bs)]
        run_batch([(r, encode_record(processor.tokenizer, r, processor=processor)) for r in rs])
    queue = asyncio.Queue()
    worker_task = asyncio.create_task(worker())             # kept: /health reports it, and an unreferenced task can be garbage-collected
    print(f"warm and ready (quant={QUANT}{"/" + ACT_SCALE + ("/packed" if USE_PACKED else "") if QUANT == "nvfp4" else ""}, gpu_alloc={torch.cuda.memory_allocated()/1e9:.1f}GB)", flush=True)


async def infer(req: dict) -> dict:
    """Request dict in the model's own format (openai_compat.Plan.request) -> its response dict. ValueError = bad request."""
    loop = asyncio.get_running_loop(); t = time.perf_counter(); STATS["requests"] += 1
    try:
        r, enc = await loop.run_in_executor(encode_pool, prepare, req)
        fut = loop.create_future()
        await queue.put((r, enc, fut))
        out = await fut
    except Exception:
        STATS["errors"] += 1; raise
    STATS["latency_s"] += time.perf_counter() - t; STATS["input_tokens"] += out["usage"]["input_tokens"]
    return out


@app.get("/health")
def health():
    ok = worker_task is not None and not worker_task.done()   # a finished worker means requests would queue forever
    return JSONResponse({"ok": ok, "quant": QUANT}, status_code=200 if ok else 503)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = [f"clef_{k}_total {v}" for k, v in STATS.items()]
    lines.append(f"clef_queue_depth {queue.qsize() if queue is not None else 0}")
    lines.append(f"clef_gpu_allocated_bytes {torch.cuda.memory_allocated()}")
    return "\n".join(lines) + "\n"


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "cloudflare"}]}


def oai_error(status, message, kind="invalid_request_error"):
    return JSONResponse({"error": {"message": message, "type": kind, "param": None, "code": None}}, status_code=status)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    import openai_compat as OC
    try:
        body = await request.json(); plan = OC.to_native(body)
    except OC.Unsupported as ex: return oai_error(400, str(ex))
    except Exception as ex: return oai_error(400, f"malformed request: {ex}")
    try:
        native = await infer(plan.request)
    except HTTPException as ex: return oai_error(ex.status_code, ex.detail, "server_error")
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as ex: return oai_error(400, str(ex) or type(ex).__name__)
    resp = OC.to_openai(plan, native, body.get("model") or MODEL_NAME)
    if not body.get("stream"): return resp
    return StreamingResponse(OC.sse(resp, include_usage=bool((body.get("stream_options") or {}).get("include_usage"))), media_type="text/event-stream")
