"""Quantize Clef-Flash to NVFP4 once and write a self-contained model directory (default clef-flash-nvfp4/):
model_nvfp4.safetensors + the release's config, tokenizer, processor, head config, model code and licence. Then check that the packed
model, loaded from disk, gives the same outputs as the freshly quantized one."""
import sys, json, shutil, time, torch
from pathlib import Path
import clef_fast as CF
SRC = Path("clef-flash"); DST = Path(sys.argv[1] if len(sys.argv) > 1 else "clef-flash-nvfp4")
if not (SRC / "config.json").exists():                    # the original bf16 release, 19 GB; only needed for this export
    from huggingface_hub import snapshot_download
    snapshot_download("Cloudflare/clef-flash", local_dir=str(SRC))
t = time.time(); model, proc = CF.load_nvfp4(str(SRC)); print(f"quantized in {time.time()-t:.0f}s", flush=True)
n_fp4, n = CF.export_packed(model, DST)
for f in SRC.iterdir():
    if f.is_file() and not f.name.startswith("model") and f.suffix != ".safetensors": shutil.copy2(f, DST / f.name)
shutil.copy2("clef_fast_calib.json", DST / "clef_fast_calib.json")
(DST / "NVFP4_README.md").write_text(f"""# Clef-Flash, NVFP4 (W4A4)

Derived from [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) (Apache-2.0, see LICENSE). Changes: the {n_fp4} language-model
linear layers are quantized to NVFP4 (4-bit E2M1 values, one FP8 E4M3 scale per 16 values, one FP32 scale per tensor) with torchao 0.18.0;
embeddings, norms, the vision tower, `lm_head` and the joint schema head are unchanged bf16. `model_nvfp4.safetensors` is not a transformers
checkpoint: load it with `clef_fast.load_packed()`. `clef_fast_calib.json` holds calibrated activation maxima for the static-scale mode.
""")
import joint_schema_model as J
def probe(m, p):
    reqs = [{"model": "c", "state": "I was charged twice on my invoice.", "questions": {"d": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "technical": "Bugs", "sales": "Demos"}}, "o": {"type": "noul", "instructions": "Is a service down?"}}},
            {"model": "c", "state": "Some filler context about the customer. " * 400, "questions": {"s": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]}}}]
    out = []
    for r in reqs:
        enc = J.encode_record(p.tokenizer, r, processor=p)
        with torch.inference_mode(): lg = m(J.collate_records([enc], p.tokenizer.pad_token_id, torch.device("cuda")))[0]
        out += [l.float().softmax(-1).cpu() for l in lg]
    return torch.cat(out)
CF.apply_fast(model); a = probe(model, proc)
del model; import gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
t = time.time(); m2, p2 = CF.load_packed(DST); CF.apply_fast(m2); dt = time.time() - t
b = probe(m2, p2)
size = sum(f.stat().st_size for f in DST.iterdir()) / 1e9
print(f"packed dir {DST}: {size:.1f} GB, {n_fp4} FP4 linears + {n - 3 * n_fp4} other tensors")
print(f"load from packed: {dt:.1f}s, GPU peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
print(f"outputs identical to the freshly quantized model: {bool(torch.equal(a, b))} (max |dp| {float((a - b).abs().max()):.2e})")
