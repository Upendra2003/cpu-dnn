"""Linear/Conv layers whose multiplications go through an approximate multiplier.

Domains (layer.domain):
  None     plain float32 PyTorch (training the float baseline)
  "float"  approximate multiplier applied directly to the float values (no quantization)
  "trunc"  floats cast to int by truncation, then multiplied -- the naive path that breaks
  8, 16    symmetric fixed point: per-tensor activation scale, per-output-channel weight scale,
           integer products accumulated exactly, then rescaled: y = sx*sw*acc + bias

When training with an approximate domain, the forward value is the approximate output and the
gradient is that of the fake-quantized float layer (straight-through estimator).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .matmul import approx_conv2d, approx_linear


def _ste_round(x):
    return x + (torch.round(x) - x).detach()


def fake_quant(x, scale, bits):
    qmax = 2 ** (bits - 1) - 1
    return torch.clamp(_ste_round(x / scale), -qmax, qmax)


class _ApproxMixin:
    def _init_approx(self):
        self.domain = None
        self.mult = "exact"
        self.register_buffer("act_max", torch.zeros(()))
        self.calibrating = False
        self.acc_max = 0.0          # max |integer accumulator| seen, for overflow analysis

    def _weight_scale(self, bits):
        qmax = 2 ** (bits - 1) - 1
        wmax = self.weight.detach().abs().flatten(1).amax(1).clamp_min(1e-12)
        return wmax / qmax

    def _forward(self, x, float_op, approx_op):
        if self.calibrating:
            self.act_max = torch.maximum(self.act_max, x.detach().abs().max())
        if self.domain is None:
            return float_op(x, self.weight, self.bias)

        shape = (1, -1) + (1,) * (x.dim() - 2)
        if self.domain == "float":
            xi, wi, scale = x, self.weight, None
        elif self.domain == "trunc":
            xi, wi, scale = torch.trunc(x), torch.trunc(self.weight), None
        else:
            bits = self.domain
            sx = (self.act_max / (2 ** (bits - 1) - 1)).clamp_min(1e-12)
            sw = self._weight_scale(bits)
            xi = fake_quant(x, sx, bits)
            wi = fake_quant(self.weight, sw.view(-1, *([1] * (self.weight.dim() - 1))), bits)
            scale = (sx * sw).view(shape)

        # Reference path (carries the gradient): exact products of the same operands.
        ref = float_op(xi, wi, None)
        with torch.no_grad():
            acc = approx_op(xi.detach().double(), wi.detach().double(), self.mult,
                            integer=isinstance(self.domain, int)).float()
            if isinstance(self.domain, int):
                self.acc_max = max(self.acc_max, acc.abs().max().item())
        out = ref + (acc - ref).detach()
        if scale is not None:
            out = out * scale
        return out + self.bias.view(shape)


class ALinear(nn.Linear, _ApproxMixin):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._init_approx()

    def forward(self, x):
        return self._forward(x, F.linear, approx_linear)


class AConv2d(nn.Conv2d, _ApproxMixin):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._init_approx()

    def forward(self, x):
        s, p = self.stride[0], self.padding[0]
        return self._forward(
            x,
            lambda x, w, b: F.conv2d(x, w, b, stride=s, padding=p),
            lambda x, w, m, integer: approx_conv2d(x, w, m, stride=s, padding=p, integer=integer),
        )


def approx_layers(model):
    return [m for m in model.modules() if isinstance(m, _ApproxMixin)]


def configure(model, domain, mult):
    for m in approx_layers(model):
        m.domain, m.mult, m.acc_max = domain, mult, 0.0


@torch.no_grad()
def calibrate(model, x, batch=1000):
    """Record the max |input| of every layer over x (float forward)."""
    configure(model, None, "exact")
    for m in approx_layers(model):
        m.act_max.zero_()
        m.calibrating = True
    model.eval()
    for i in range(0, len(x), batch):
        model(x[i:i + batch])
    for m in approx_layers(model):
        m.calibrating = False


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            ALinear(784, 256), nn.ReLU(),
            ALinear(256, 128), nn.ReLU(),
            ALinear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            AConv2d(1, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            AConv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            ALinear(400, 120), nn.ReLU(),
            ALinear(120, 84), nn.ReLU(),
            ALinear(84, 10),
        )

    def forward(self, x):
        return self.net(x)


MODELS = {"mlp": MLP, "lenet5": LeNet5}


def count_macs(model, input_shape=(1, 1, 28, 28)):
    """Multiplications per inference, per approximate layer."""
    macs = []
    hooks = []

    def hook(m, inp, out):
        k = m.weight[0].numel()
        macs.append(out[0].numel() * k)

    for m in approx_layers(model):
        hooks.append(m.register_forward_hook(hook))
    configure(model, None, "exact")
    with torch.no_grad():
        model(torch.zeros(input_shape, device=next(model.parameters()).device))
    for h in hooks:
        h.remove()
    return macs
