"""Clef-Flash backbone in NVFP4 (torchao) with the slow eager pieces fused:
the FP4 GEMMs are fast, but the stock path spends most of a long request in unfused elementwise work around them.

  load_nvfp4(path)   bf16 checkpoint -> language-model linears quantized to NVFP4 (W4A4, dynamic activation scale)
  export_packed / load_packed   save the quantized model once (export_nvfp4.py) and load it without the bf16 checkpoint
  apply_fast(model)  fused RMSNorm / gated RMSNorm / silu*mul, FP4 linears that quantize an input once for every projection reading it
                     and fold the output scale into the next fused kernel instead of a separate pass
"""
import sys, types
import torch, torch.nn as nn, torch.nn.functional as F
from torchao.prototype.mx_formats.kernels import mslk_quantize_nvfp4
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
from torchao.prototype.mx_formats.utils import to_blocked
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

SCALE_DIV = 448.0 * 6.0                                    # torchao: per-tensor scale = amax / (F8E4M3_MAX * F4_E2M1_MAX)


def nvfp4_filter(m, fqn):
    return isinstance(m, nn.Linear) and "visual" not in fqn and "lm_head" not in fqn and m.in_features % 128 == 0 and m.out_features % 16 == 0


def load_nvfp4(path="clef-flash", device="cuda"):
    """Load on the CPU and quantize one linear at a time on the GPU. Loading bf16 straight to the GPU and quantizing in place leaves the allocator
    holding ~20 GB for 9 GB of tensors (the freed bf16 blocks are pinned by small quantized tensors), which breaks a per-process memory cap."""
    if path not in sys.path: sys.path.insert(0, path)
    import joint_schema_model as J
    from torchao.quantization import quantize_
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
    model, proc = J.load_release_model(path, device="cpu")
    cfg = NVFP4DynamicActivationNVFP4WeightConfig(); lm = model.language_model
    for name, mod in list(lm.named_modules()):
        if not nvfp4_filter(mod, name): continue
        assert mod.bias is None
        new = nn.Linear(mod.in_features, mod.out_features, bias=False, device=device, dtype=mod.weight.dtype)
        new.weight = nn.Parameter(mod.weight.data.to(device), requires_grad=False)
        box = nn.Sequential(new); quantize_(box, cfg, filter_fn=lambda m, fqn: isinstance(m, nn.Linear))
        parent, _, attr = name.rpartition("."); setattr(lm.get_submodule(parent) if parent else lm, attr, box[0])
        mod.weight = None
    done = {id(m) for m in lm.modules() if isinstance(m, nn.Linear) and isinstance(m.weight, NVFP4Tensor)}
    for mod in model.modules():                             # everything else (embeddings, norms, vision tower, head, small linears) as is
        if id(mod) in done: continue
        for k, p in list(mod._parameters.items()):
            if p is not None: mod._parameters[k] = nn.Parameter(p.data.to(device), requires_grad=False)
        for k, b in list(mod._buffers.items()):
            if b is not None: mod._buffers[k] = b.to(device)
    torch.cuda.empty_cache()
    return model.eval(), proc


PACKED_FILE = "model_nvfp4.safetensors"


def export_packed(model, out_dir):
    """Write the quantized model as one safetensors file: packed FP4 linears + every other parameter and buffer as is."""
    from pathlib import Path
    from safetensors.torch import save_file
    tensors = {}; fp4 = {name for name, mod in model.named_modules() if is_fp4(mod)}
    for name in sorted(fp4):
        q, sc, pts = packed_parts(model.get_submodule(name))
        tensors[name + ".fp4_qdata"], tensors[name + ".fp4_scale"], tensors[name + ".fp4_pts"] = q.contiguous(), sc.contiguous(), pts.reshape(1)
    for kind, items in (("param", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, t in items:
            if name.rpartition(".")[0] in fp4: continue
            tensors[name] = t.detach().contiguous()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_file({k: v.cpu() for k, v in tensors.items()}, str(Path(out_dir) / PACKED_FILE), metadata={"format": "clef-nvfp4-v1", "n_fp4": str(len(fp4))})
    return len(fp4), len(tensors)


def load_packed(path, device="cuda"):
    """Build the model straight from a pre-quantized directory (export_packed + the release's config / tokenizer / head files).
    No bf16 weights are read and nothing is quantized at load."""
    import json
    from pathlib import Path
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
    path = Path(path)
    if str(path) not in sys.path: sys.path.insert(0, str(path))
    import joint_schema_model as J
    cfg = AutoConfig.from_pretrained(path)
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            backbone = Qwen3_5ForConditionalGeneration(cfg)
            head = J.JointSchemaHead(**json.loads((path / "joint_head_config.json").read_text()))
    finally: torch.set_default_dtype(torch.float32)
    backbone.config.use_cache = False
    model = J.ClefModel(backbone, head)
    with safe_open(str(path / PACKED_FILE), "pt", device=str(device)) as f:
        keys = set(f.keys())
        for name in sorted(k[:-len(".fp4_qdata")] for k in keys if k.endswith(".fp4_qdata")):
            new = PackedLinear(f.get_tensor(name + ".fp4_qdata"), f.get_tensor(name + ".fp4_scale"), f.get_tensor(name + ".fp4_pts").reshape(()))
            parent, _, attr = name.rpartition("."); setattr(model.get_submodule(parent), attr, new)
        for name in keys:
            if ".fp4_" in name: continue
            mod_path, _, attr = name.rpartition("."); mod = model.get_submodule(mod_path); t = f.get_tensor(name)
            if attr in mod._parameters: mod._parameters[attr] = nn.Parameter(t, requires_grad=False)
            else: mod._buffers[attr] = t
    meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == "meta"]
    assert not meta, f"not in the packed checkpoint: {meta[:5]}"
    return model.eval(), AutoProcessor.from_pretrained(path)


@torch.compile(dynamic=True)
def _rmsnorm(x, w, eps: float):                             # Qwen3.5: x / rms(x) * (1 + w), fp32 math, one kernel
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).to(x.dtype)


@torch.compile(dynamic=True)
def _rmsnorm_gated(x, gate, w, eps: float):
    xf = x.float()
    n = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    return ((w * n).float() * F.silu(gate.float())).to(x.dtype)


@torch.compile(dynamic=True)
def _silu_mul(g, u, ag, au):                                # silu(g * ag) * (u * au): the FP4 output scales ride along
    return (F.silu(g.float() * ag) * (u.float() * au)).to(g.dtype)


@torch.compile(dynamic=True)
def _scale(x, a):
    return (x.float() * a).to(x.dtype)


@torch.compile(dynamic=True)
def _act_scale(x):                                          # dynamic per-tensor activation scale, one reduction kernel
    return x.abs().amax().float() / SCALE_DIV


class PackedLinear(nn.Module):
    """An NVFP4 linear loaded from a pre-quantized checkpoint: packed FP4 data [N, K/2], swizzled e4m3 block scales, per-tensor scale.
    It is only ever called through the fused forwards installed by apply_fast()."""
    def __init__(self, qdata, scale, pts):
        super().__init__()
        self.register_buffer("qdata", qdata); self.register_buffer("scale", scale); self.register_buffer("pts", pts)
        self.out_features, self.in_features = qdata.shape[0], qdata.shape[1] * 2; self.bias = None


def is_fp4(mod):
    return isinstance(mod, PackedLinear) or (isinstance(mod, nn.Linear) and isinstance(mod.weight, NVFP4Tensor))


def packed_parts(lin):
    """(qdata uint8 [N, K/2], swizzled block scales uint8, per-tensor scale fp32) of a torchao-quantized or packed linear."""
    if isinstance(lin, PackedLinear): return lin.qdata, lin.scale, lin.pts
    qt = lin.weight; assert lin.bias is None
    N, K = qt.shape
    sc = qt.scale if qt.is_swizzled_scales else to_blocked(qt.scale.view(N, K // 16))
    pts = qt.per_tensor_scale.float() if qt.per_tensor_scale is not None else torch.ones((), device=qt.qdata.device)
    return qt.qdata.view(torch.uint8), sc.contiguous().view(torch.uint8), pts.reshape(())


class Fp4W:
    """The weight side of one NVFP4 linear: packed data + swizzled block scales + per-tensor scale."""
    def __init__(self, lin):
        qdata, scale, pts = packed_parts(lin)
        self.wq = qdata.view(torch.float4_e2m1fn_x2).t(); self.ws = scale.view(torch.float8_e4m3fn); self.wpts = pts
        self.out_features = qdata.shape[0]

    def mm(self, xq):                                       # xq = quant(x); returns the unscaled GEMM output and the scalar to multiply by
        xd, xs, pts = xq
        return torch._scaled_mm(xd, self.wq, xs, self.ws, out_dtype=torch.bfloat16), pts * self.wpts


class Act:
    """Activation-scale policy.
      mode "dynamic": per-tensor scale from the tensor's own amax on every call (torchao's default; a full extra pass over the input, 16% of GPU time at 3.6k tokens)
      mode "calib"  : dynamic, and record the largest amax seen per site
      mode "static" : scale fixed per site from calibration (largest amax x HEADROOM); inputs above that saturate
    A "site" is one distinct input tensor (gate/up share one; q/k/v share one; ...)."""
    mode = "dynamic"; HEADROOM = 2.0
    last = (None, None)                                     # (input tensor, its quantized form): projections reading the same tensor quantize it once


def quant(x2, site):
    if Act.last[0] is x2:                                   # same input as the previous projection: same amax, same quantized tensor
        if Act.mode == "calib": site["amax"] = max(site.get("amax", 0.0), float(Act.last[1][2]) * SCALE_DIV)
        return Act.last[1]
    if Act.mode == "static": pts = site["pts"]
    else:
        pts = _act_scale(x2)
        if Act.mode == "calib": site["amax"] = max(site.get("amax", 0.0), float(pts) * SCALE_DIV)
    xs, xd = mslk_quantize_nvfp4(x2, pts)
    q = (xd.view(torch.float4_e2m1fn_x2), xs.view(torch.float8_e4m3fn), pts); Act.last = (x2, q)
    return q


def _mlp_forward(self, x):
    shape = x.shape; x2 = x.reshape(-1, shape[-1]).contiguous()
    xq = quant(x2, self._site_in)                           # gate and up read the same input: quantize it once
    (g, ag), (u, au) = self._g.mm(xq), self._u.mm(xq)
    d, ad = self._d.mm(quant(_silu_mul(g, u, ag, au), self._site_mid))
    Act.last = (None, None)
    return _scale(d, ad).reshape(*shape[:-1], -1)


def _lin_forward(self, x):
    shape = x.shape; x2 = x if x.dim() == 2 and x.is_contiguous() else _view2(x)
    out, a = self._w.mm(quant(x2, self._site))
    return _scale(out, a).reshape(*shape[:-1], -1)


_views = {}
def _view2(x):
    """2-D view of x, the same object for the same x: q/k/v (and the linear-attention input projections) are called with one tensor,
    and the quantization cache keys on identity."""
    hit = _views.get("v")
    if hit is not None and hit[0] is x: return hit[1]
    v = x.reshape(-1, x.shape[-1]).contiguous(); _views["v"] = (x, v)
    return v


def sites(model):
    """name -> site dict, for saving / loading calibration."""
    out = {}
    for name, mod in model.language_model.named_modules():
        for attr in ("_site", "_site_in", "_site_mid"):
            if attr in mod.__dict__: out[f"{name}.{attr}"] = mod.__dict__[attr]
    return out


def set_static(model, amax: dict):
    dev = next(model.parameters()).device
    for k, site in sites(model).items():
        site["pts"] = torch.tensor(amax[k] * Act.HEADROOM / SCALE_DIV, dtype=torch.float32, device=dev)
    Act.mode = "static"


def apply_fast(model, norms=True, mlp=True, linears=True):
    lm = model.language_model
    if norms:
        M.Qwen3_5RMSNorm.forward = lambda self, x: _rmsnorm(x, self.weight, self.eps)
        M.Qwen3_5RMSNormGated.forward = lambda self, h, gate: _rmsnorm_gated(h, gate, self.weight, self.variance_epsilon)
    done = set()
    if mlp:
        for name, mod in lm.named_modules():
            if isinstance(mod, M.Qwen3_5MLP) and all(is_fp4(getattr(mod, n)) for n in ("gate_proj", "up_proj", "down_proj")):
                mod._g, mod._u, mod._d = (Fp4W(getattr(mod, n)) for n in ("gate_proj", "up_proj", "down_proj")); mod._site_in, mod._site_mid = {}, {}
                mod.forward = types.MethodType(_mlp_forward, mod); done.update(id(getattr(mod, n)) for n in ("gate_proj", "up_proj", "down_proj"))
    if linears:
        for name, mod in lm.named_modules():
            if is_fp4(mod) and id(mod) not in done:
                mod._w = Fp4W(mod); mod._site = {}; mod.forward = types.MethodType(_lin_forward, mod)
    return model
