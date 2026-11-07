"""The integer layer program: one description of a quantized network, used three ways.

  QNet          quantization-aware training on the GPU (fake quantization, approximate products, STE)
  export()      the integer program: int weights, per-channel fixed-point rescale (M, C, SH), buffer shapes
  int_forward() bit-exact integer reference of what hw/sw/tiny.c computes on Ibex

Number formats ("bits"): weights are symmetric signed with QW levels per side; activations come out of ReLU,
so they are unsigned 0..QA (an int8 activation is a uint8 0..255). The network input is either the 8-bit pixel
itself (prog["input"] == "pixels"; int4: rounded to 0..15) or a signed feature such as an MFCC, quantized
symmetrically to -QW..QW with a calibrated scale ("signed").

Methods: how a layer's sum of products is computed.
  exact   exact products                      dn1, dn2   RoBA-Dn1 / RoBA-Dn2 products (sign-magnitude)
  sqtab   exact, via a table of squares: sum xw = (sum S(x+w) - sum S(x) - sum S(w)) / 2   (same numbers as exact)
  sqapx   the same identity with approximate squares S~(v) = (|v| << (k+1)) - (1 << 2k), k = LOD(|v|)

Layer output, for output channel o (all integer, as on Ibex):
  y = (acc * M[o] + skip * MS[o] + C[o] + 2^(SH[o]-1)) >> SH[o],  then clamp to 0..QA (ReLU)
where skip is the block input (identity shortcut) or the accumulator of a 1x1 projection conv.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from roba.multipliers import factors

BITS = (4, 8, 16)
QW = {4: 7, 8: 127, 16: 32767}
QA = {4: 15, 8: 255, 16: 65535}
METHODS = ("exact", "sqtab", "dn1", "dn2", "sqapx")
ROBA = {"exact": "exact", "sqtab": "exact", "dn1": "roba_dn1", "dn2": "roba_dn2"}
TRAINED_WITH = {"exact": "exact", "sqtab": "exact", "dn1": "dn1", "dn2": "dn2", "sqapx": "exact"}
M_BITS = {4: 16, 8: 16, 16: 20}          # precision of the fixed-point rescale factor (int16: |acc| M < 2^62)


def feasible(bits, method):
    """sqtab at int16 needs a 131k-entry table (512 kB): reported as not feasible. sqapx is the negative
    result from the MNIST study and is only run at int8."""
    if method == "sqtab":
        return bits <= 8
    if method == "sqapx":
        return bits == 8
    return True


def qa_in(bits):
    return min(QA[bits], 255)


def pad2(p):
    return (p, p) if isinstance(p, int) else tuple(p)


def k2(k):
    return (k, k) if isinstance(k, int) else tuple(k)


# ---------------------------------------------------------------- fold batch norm into a float program

@torch.no_grad()
def fold(model):
    """model.spec() -> list of ops with float w (O, C, k, k), b (O), stride, pad. BN is folded into the conv;
    a projection conv's bias moves to the conv that adds its accumulator; fc becomes a 1x1 conv."""
    ops, moved_bias = [], {}
    for s in model.spec():
        if s["op"] == "conv":
            c = s["conv"]
            w = c.weight.detach().double()
            b = c.bias.detach().double() if c.bias is not None else torch.zeros(w.shape[0], dtype=torch.float64,
                                                                                   device=w.device)
            if s["bn"] is not None:
                bn = s["bn"]
                g = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
                w = w * g.view(-1, 1, 1, 1)
                b = bn.bias.double() + (b - bn.running_mean.double()) * g
            op = {"op": "conv", "name": s["name"], "src": s["src"], "dst": s["dst"], "w": w, "b": b,
                  "stride": c.stride[0], "pad": pad2(c.padding), "groups": c.groups, "relu": s["relu"],
                  "skip": s["skip"]}
            if s["relu"] is None:                                 # accumulator output: bias goes to its consumer
                moved_bias[s["dst"]] = b
                op["b"] = torch.zeros_like(b)
            if s["skip"] is not None and s["skip"]["kind"] == "acc":
                op["b"] = op["b"] + moved_bias[s["skip"]["src"]]
            ops.append(op)
        elif s["op"] == "avgpool":
            ops.append(dict(s))
        elif s["op"] == "fc":                                  # a 1x1 conv on a 1x1 map; last one gives logits
            fc = s["fc"]
            w, b = fc.weight.detach().double()[:, :, None, None], fc.bias.detach().double()
            if s.get("bn") is not None:
                bn = s["bn"]
                g = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
                w = w * g.view(-1, 1, 1, 1)
                b = bn.bias.double() + (b - bn.running_mean.double()) * g
            ops.append({"op": "conv", "name": s["name"], "src": s["src"], "dst": s["dst"], "w": w, "b": b,
                        "stride": 1, "pad": 0, "relu": s.get("relu", "logits"), "skip": None})
    return ops


def float_forward(ops, x, record=None):
    """Float run of a folded program on the float model input; record[name] collects every activation buffer."""
    v = {"in": x.double()}
    for op in ops:
        src = v[op["src"]]
        if op["op"] == "avgpool":
            out = F.avg_pool2d(src, op["k"])
        else:
            out = F.conv2d(src, op["w"], op["b"], op["stride"], op["pad"], 1, op.get("groups", 1))
            if op["relu"] is None:
                v[op["dst"]] = out
                continue
            if op["skip"] is not None:
                out = out + v[op["skip"]["src"]]
            if op["relu"] == "logits":
                return out.flatten(1)
            out = F.relu(out)
        v[op["dst"]] = out
        if record is not None:
            record.setdefault(op["dst"], []).append(out.flatten())


@torch.no_grad()
def calibrate(ops, x, pct, batch=250, keep=200_000, to_float=lambda x: x.double() / 255):
    """Activation scale per buffer: the pct-th percentile of its positive values over x (a fixed random sample
    of at most `keep` values per batch, to bound memory), divided by QA later. to_float maps a batch of stored
    inputs to the float model's input (pixels / 255 by default)."""
    gen = torch.Generator(device=x.device).manual_seed(0)
    rec = {}
    for i in range(0, len(x), batch):
        r = {}
        float_forward(ops, to_float(x[i:i + batch]).double(), r)
        for k, vs in r.items():
            v = vs[0].float()
            v = v[v > 0]
            if v.numel() > keep:                       # keep the maximum so that pct = 100 stays exact
                v = torch.cat([v[torch.randint(0, v.numel(), (keep,), device=v.device, generator=gen)], v.max()[None]])
            rec.setdefault(k, []).append(v)
    out = {}
    for k, vs in rec.items():
        v = torch.cat(vs)
        out[k] = float(torch.quantile(v, pct / 100)) if pct < 100 else float(v.max())
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- products of a conv layer

def _ste(x, xr):
    return x + (xr - x).detach()


def factor_conv(x, w, mult, stride, pad, groups=1):
    """sum of exact convs of per-operand factors = conv with every product replaced by `mult`."""
    if mult == "exact":
        return F.conv2d(x, w, None, stride, pad, 1, groups)
    fx, fw = factors(x, mult, "a"), factors(w, mult, "b")
    return sum(F.conv2d(a, b, None, stride, pad, 1, groups) for a, b in zip(fx, fw))


def sq_apx(v):
    """Approximate square of an integer tensor (int64): 2^(k+1)|v| - 2^(2k) with k = LOD(|v|); 0 for 0."""
    a = v.abs()
    k = (torch.frexp(a.double())[1] - 1).clamp_min(0).long()
    return torch.where(a > 0, (2 * a - (torch.ones_like(a) << k)) << k, torch.zeros_like(a))


def sqapx_conv(x, w, stride, pad, groups=1, chunk=1 << 24):
    """Integer accumulator of the approximate-square method. x, w: int64."""
    n, _, h, wd = x.shape
    o, cg, kh, kw = w.shape
    ph, pw = pad2(pad)
    cols = F.unfold(x.double(), (kh, kw), padding=(ph, pw), stride=stride).round().long()   # (N, C*kh*kw, L)
    oh = (h + 2 * ph - kh) // stride + 1
    ow = (wd + 2 * pw - kw) // stride + 1
    L, kg = cols.shape[2], cg * kh * kw
    cols = cols.view(n, groups, kg, L).permute(0, 3, 1, 2).reshape(n * L, groups, kg)       # (N*L, G, Kg)
    wf = w.reshape(groups, o // groups, kg)                                                 # (G, Og, Kg)
    rows = max(1, chunk // (o * kg))
    d = torch.cat([sq_apx(cols[i:i + rows, :, None, :] + wf[None]).sum(-1) for i in range(0, len(cols), rows)])
    d = d - sq_apx(cols).sum(-1)[:, :, None] - sq_apx(wf).sum(-1)[None]
    return (d >> 1).reshape(n, L, o).transpose(1, 2).reshape(n, o, oh, ow)


# ---------------------------------------------------------------- quantization-aware training

class QNet(nn.Module):
    """Fake-quantized network for one (bits, training method). Weights are learned; activation scales are fixed
    by calibration. Forward values use the method's approximate products, gradients the exact ones (STE)."""

    def __init__(self, ops, scales, bits, method, signed_in=None):
        """signed_in: None for pixel input, else the float scale of the signed network input (before /QW)."""
        super().__init__()
        self.ops = [{k: v for k, v in op.items() if k not in ("w", "b")} for op in ops]
        self.w = nn.ParameterDict({op["name"]: nn.Parameter(op["w"].float()) for op in ops if op["op"] == "conv"})
        self.b = nn.ParameterDict({op["name"]: nn.Parameter(op["b"].float()) for op in ops if op["op"] == "conv"})
        self.bits, self.method = bits, method
        self.mult = ROBA[TRAINED_WITH[method]]
        self.s = dict(scales)                                           # buffer -> scale (float)
        self.signed_in = signed_in is not None
        self.s["in"] = signed_in / QW[bits] if self.signed_in else 1.0 / qa_in(bits)

    def wq(self, name):
        w = self.w[name]
        sw = (w.detach().abs().amax((1, 2, 3)) / QW[self.bits]).clamp_min(1e-12)
        wi = torch.clamp(_ste(w / sw.view(-1, 1, 1, 1), torch.round(w / sw.view(-1, 1, 1, 1))),
                         -QW[self.bits], QW[self.bits])
        return wi, sw

    def act(self, v, buf):
        s, q = self.s[buf], QA[self.bits]
        return torch.clamp(_ste(v / s, torch.round(v / s)), 0, q) * s

    def forward(self, x):
        """x: images (uint8, or float pixels / 255), or float signed features, on the GPU."""
        if self.signed_in:
            q = QW[self.bits]
            v = {"in": torch.clamp(torch.round(x.float() / self.s["in"]), -q, q) * self.s["in"]}
        else:                                       # uint8 pixels, or pixels / 255 (the float model's input)
            xf = x.float() / 255 if x.dtype == torch.uint8 else x.float()
            v = {"in": torch.round(xf * qa_in(self.bits)) * self.s["in"]}
        for op in self.ops:
            src = v[op["src"]]
            if op["op"] == "avgpool":
                v[op["dst"]] = self.act(F.avg_pool2d(src, op["k"]), op["dst"])
                continue
            sx = self.s[op["src"]]
            xi = _ste(src / sx, torch.round(src / sx))
            wi, sw = self.wq(op["name"])
            g = op.get("groups", 1)
            ref = F.conv2d(xi, wi, None, op["stride"], op["pad"], 1, g)
            if self.mult != "exact":
                with torch.no_grad():
                    acc = factor_conv(xi.detach(), wi.detach(), self.mult, op["stride"], op["pad"], g)
                ref = _ste(ref, acc)
            out = ref * (sx * sw).view(1, -1, 1, 1)
            if op["relu"] is None:
                v[op["dst"]] = out
                continue
            out = out + self.b[op["name"]].view(1, -1, 1, 1)
            if op["skip"] is not None:
                out = out + v[op["skip"]["src"]]
            if op["relu"] == "logits":
                return out.flatten(1)
            v[op["dst"]] = self.act(F.relu(out), op["dst"])


# ---------------------------------------------------------------- export to the integer program

def _fixed(r, mbits):
    """Positive real r -> (M, SH) with M = round(r * 2^SH) in [2^(mbits-1), 2^mbits)."""
    sh = mbits - 1 - math.floor(math.log2(r))
    m = round(r * 2 ** sh)
    if m >= 2 ** mbits:
        sh, m = sh - 1, round(r * 2 ** (sh - 1))
    return m, sh


@torch.no_grad()
def export(q, shapes):
    """QNet -> integer program (CPU tensors). shapes: buffer -> (C, H, W)."""
    bits, prog = q.bits, []
    mb = M_BITS[bits]
    for op in q.ops:
        if op["op"] == "avgpool":
            r = q.s[op["src"]] / (k2(op["k"])[0] * k2(op["k"])[1] * q.s[op["dst"]])
            m, sh = _fixed(r, mb)
            prog.append({**op, "M": m, "SH": sh, "shape_in": shapes[op["src"]], "shape_out": shapes[op["dst"]]})
            continue
        wi, sw = q.wq(op["name"])
        wi, sw = wi.round().long().cpu(), sw.double().cpu()
        sx = q.s[op["src"]]
        P = {**op, "wq": wi, "shape_in": shapes[op["src"]], "shape_out": shapes[op["dst"]]}
        if op["relu"] is None:
            P["acc_scale"] = (sx * sw)                               # float value of one accumulator unit
            prog.append(P)
            continue
        if op["relu"] == "logits":
            s_out = float((sx * sw).max()) / 256
        else:
            s_out = q.s[op["dst"]]
        r = (sx * sw / s_out).double()
        rs = None
        if op["skip"] is not None:
            sk = op["skip"]
            rs = (torch.full_like(r, q.s[sk["src"]] / s_out) if sk["kind"] == "identity"
                  else next(p for p in prog if p["dst"] == sk["src"])["acc_scale"] / s_out)
        b = q.b[op["name"]].detach().double().cpu() / s_out
        # |acc| M + |skip| MS + |C| must stay below 2^62 (int64 on Ibex). Products are at most QA*QW; the table
        # methods (int8 and below) can reach K (QA + QW)^2 before halving. A channel's shift is lowered only
        # when this worst case needs it.
        K = wi[0].numel()
        acc_max = K * QA[bits] * QW[bits] if bits > 8 else K * (QA[bits] + 2 * QW[bits]) ** 2
        skip_max = 0
        if rs is not None:
            skip_max = QA[bits] if op["skip"]["kind"] == "identity" else \
                next(p for p in prog if p["dst"] == op["skip"]["src"])["wq"][0].numel() * QA[bits] * QW[bits]
        M, SH, MS, C = [], [], [], []
        for o in range(len(r)):
            m, sh = _fixed(float(r[o]), mb)
            while True:
                ms = round(float(rs[o]) * 2 ** sh) if rs is not None else 0
                c = round(float(b[o]) * 2 ** sh)
                if acc_max * m + skip_max * ms + abs(c) + 2 ** sh < 2 ** 62:
                    break
                sh -= 1
                m = round(float(r[o]) * 2 ** sh)
            assert m >= 2 ** 10, f"{op['name']}: rescale factor too coarse"
            M.append(m), SH.append(sh), MS.append(ms), C.append(c)
        P["M"], P["SH"], P["C"] = torch.tensor(M), torch.tensor(SH), torch.tensor(C)
        if rs is not None:
            P["MS"] = torch.tensor(MS)
        P["s_out"] = s_out
        prog.append(P)
    return {"bits": bits, "method": q.method, "ops": prog, "s_in": q.s["in"],
            "input": "signed" if q.signed_in else "pixels"}


def buffer_shapes(ops, x):
    """Run the float program on one input to get (C, H, W) of every buffer."""
    shapes = {"in": tuple(x.shape[1:])}
    v = {"in": x.double()}
    for op in ops:
        src = v[op["src"]]
        if op["op"] == "avgpool":
            out = F.avg_pool2d(src, op["k"])
        else:
            out = F.conv2d(src, op["w"], op["b"], op["stride"], op["pad"], 1, op.get("groups", 1))
        v[op["dst"]] = out
        shapes[op["dst"]] = tuple(out.shape[1:])
    return shapes


# ---------------------------------------------------------------- bit-exact integer reference

def quantize_input(x_uint8, bits, prog=None):
    """Network input as integers. Pixels: the 8-bit value (identical for int8/int16; int4 rounds to 0..15).
    Signed features (prog["input"] == "signed"): round(x / s_in) clamped to -QW..QW."""
    if prog is not None and prog.get("input") == "signed":
        return torch.clamp(torch.round(x_uint8.double() / prog["s_in"]), -QW[bits], QW[bits]).long()
    x = x_uint8.long()
    q = qa_in(bits)
    return x if q == 255 else (x * (2 * q) + 255) // 510                # round(x * q / 255)


def int_conv(x, w, method, stride, pad, groups=1):
    """Integer accumulator (int64) of one conv layer. Products are exact integers below 2^53 in float64."""
    pad = pad2(pad)
    if method == "sqapx":
        return sqapx_conv(x, w, stride, pad, groups)
    return factor_conv(x.double(), w.double(), ROBA[method], stride, pad, groups).round().long()


@torch.no_grad()
def int_forward(prog, x_uint8, method=None):
    """Logits (int64) of the integer program; method defaults to the program's own."""
    method = method or prog["method"]
    dev = x_uint8.device
    qa = QA[prog["bits"]]
    v = {"in": quantize_input(x_uint8, prog["bits"], prog)}
    for op in prog["ops"]:
        src = v[op["src"]]
        if op["op"] == "avgpool":
            kh, kw = k2(op["k"])
            s = (F.avg_pool2d(src.double(), (kh, kw)) * (kh * kw)).round().long()
            v[op["dst"]] = ((s * op["M"] + (1 << (op["SH"] - 1))) >> op["SH"]).clamp(0, qa)
            continue
        acc = int_conv(src, op["wq"].to(dev), method, op["stride"], op["pad"], op.get("groups", 1))
        if op["relu"] is None:
            v[op["dst"]] = acc
            continue
        ch = lambda t: t.to(dev).view(1, -1, 1, 1)
        y = acc * ch(op["M"]) + ch(op["C"]) + (torch.ones_like(ch(op["SH"])) << (ch(op["SH"]) - 1))
        if op["skip"] is not None:
            y = y + v[op["skip"]["src"]] * ch(op["MS"])
        y = y >> ch(op["SH"])
        if op["relu"] == "logits":
            return y.flatten(1)
        v[op["dst"]] = y.clamp(0, qa)
