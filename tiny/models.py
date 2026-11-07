"""Float models with batch norm, following the MLPerf Tiny reference models (trained here in PyTorch).

Each model also describes itself as a list of layer specs (`spec()`) that tiny/qnet.py folds into an integer
layer program: conv layers (BN folded into weights and bias), residual adds, average pooling, fully connected.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet8(nn.Module):
    """MLPerf Tiny image classification (CIFAR-10): 3 residual stacks of 16/32/64 channels, ~78k parameters,
    ~12.5M multiply-accumulates per image."""

    def __init__(self, classes=10):
        super().__init__()
        c = nn.Conv2d
        self.conv1, self.bn1 = c(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16)
        self.s1a, self.b1a = c(16, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16)
        self.s1b, self.b1b = c(16, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16)
        self.s2a, self.b2a = c(16, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32)
        self.s2b, self.b2b = c(32, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32)
        self.s2s = c(16, 32, 1, 2, 0)                          # projection shortcut (no BN, as in the reference)
        self.s3a, self.b3a = c(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64)
        self.s3b, self.b3b = c(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64)
        self.s3s = c(32, 64, 1, 2, 0)
        self.fc = nn.Linear(64, classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.b1a(self.s1a(x)))
        x = F.relu(self.b1b(self.s1b(y)) + x)
        y = F.relu(self.b2a(self.s2a(x)))
        x = F.relu(self.b2b(self.s2b(y)) + self.s2s(x))
        y = F.relu(self.b3a(self.s3a(x)))
        x = F.relu(self.b3b(self.s3b(y)) + self.s3s(x))
        x = F.avg_pool2d(x, 8).flatten(1)
        return self.fc(x)

    def spec(self):
        """Layer program before quantization. Buffers are named; 'in' is the image."""
        L = []

        def conv(name, src, dst, conv, bn=None, relu=True, skip=None):
            L.append({"op": "conv", "name": name, "src": src, "dst": dst, "conv": conv, "bn": bn,
                      "relu": relu, "skip": skip})

        conv("conv1", "in", "x1", self.conv1, self.bn1)
        conv("s1a", "x1", "h1", self.s1a, self.b1a)
        conv("s1b", "h1", "x2", self.s1b, self.b1b, skip={"kind": "identity", "src": "x1"})
        conv("s2s", "x2", "p2", self.s2s, relu=None)            # relu=None: raw accumulator, no rescale
        conv("s2a", "x2", "h2", self.s2a, self.b2a)
        conv("s2b", "h2", "x3", self.s2b, self.b2b, skip={"kind": "acc", "src": "p2"})
        conv("s3s", "x3", "p3", self.s3s, relu=None)
        conv("s3a", "x3", "h3", self.s3a, self.b3a)
        conv("s3b", "h3", "x4", self.s3b, self.b3b, skip={"kind": "acc", "src": "p3"})
        L.append({"op": "avgpool", "name": "pool", "src": "x4", "dst": "g", "k": 8})
        L.append({"op": "fc", "name": "fc", "src": "g", "dst": "logits", "fc": self.fc})
        return L


class DSCNN(nn.Module):
    """MLPerf Tiny keyword spotting (Speech Commands, 12 classes): 49 x 10 MFCC input, a 10 x 4 stride-2 conv,
    then 4 depthwise-separable blocks of 64 channels, ~23k parameters, ~2.7M multiply-accumulates."""

    def __init__(self, classes=12, ch=64, blocks=4):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(1, ch, (10, 4), 2, (5, 1), bias=False), nn.BatchNorm2d(ch)
        self.dw = nn.ModuleList(nn.Conv2d(ch, ch, 3, 1, 1, groups=ch, bias=False) for _ in range(blocks))
        self.dwbn = nn.ModuleList(nn.BatchNorm2d(ch) for _ in range(blocks))
        self.pw = nn.ModuleList(nn.Conv2d(ch, ch, 1, bias=False) for _ in range(blocks))
        self.pwbn = nn.ModuleList(nn.BatchNorm2d(ch) for _ in range(blocks))
        self.drop1, self.drop2 = nn.Dropout(0.2), nn.Dropout(0.4)
        self.fc = nn.Linear(ch, classes)

    def forward(self, x):
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        for dw, dbn, pw, pbn in zip(self.dw, self.dwbn, self.pw, self.pwbn):
            x = F.relu(dbn(dw(x)))
            x = F.relu(pbn(pw(x)))
        x = self.drop2(F.avg_pool2d(x, x.shape[2:]).flatten(1))
        return self.fc(x)

    def spec(self):
        L = [{"op": "conv", "name": "conv1", "src": "in", "dst": "a0", "conv": self.conv1, "bn": self.bn1,
              "relu": True, "skip": None}]
        for i in range(len(self.dw)):
            L.append({"op": "conv", "name": f"dw{i}", "src": f"a{i}", "dst": f"d{i}", "conv": self.dw[i],
                      "bn": self.dwbn[i], "relu": True, "skip": None})
            L.append({"op": "conv", "name": f"pw{i}", "src": f"d{i}", "dst": f"a{i + 1}", "conv": self.pw[i],
                      "bn": self.pwbn[i], "relu": True, "skip": None})
        L.append({"op": "avgpool", "name": "pool", "src": f"a{len(self.dw)}", "dst": "g", "k": (25, 5)})
        L.append({"op": "fc", "name": "fc", "src": "g", "dst": "logits", "fc": self.fc})
        return L


class FCAutoEncoder(nn.Module):
    """MLPerf Tiny anomaly detection (ToyADMOS / DCASE 2020 ToyCar): 640 inputs (5 frames x 128 log-mel bands),
    dense 128-128-128-128-8-128-128-128-128 with batch norm and ReLU, then 640 outputs; ~264k multiply-accumulates.
    The anomaly score is the reconstruction error."""

    def __init__(self, dims=(640, 128, 128, 128, 128, 8, 128, 128, 128, 128)):
        super().__init__()
        self.fcs = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(b) for b in dims[1:])
        self.out = nn.Linear(dims[-1], dims[0])

    def forward(self, x):
        x = x.flatten(1)
        for fc, bn in zip(self.fcs, self.bns):
            x = F.relu(bn(fc(x)))
        return self.out(x)

    def spec(self):
        L, src = [], "in"
        for i, (fc, bn) in enumerate(zip(self.fcs, self.bns)):
            L.append({"op": "fc", "name": f"fc{i}", "src": src, "dst": f"h{i}", "fc": fc, "bn": bn, "relu": True})
            src = f"h{i}"
        L.append({"op": "fc", "name": "out", "src": src, "dst": "logits", "fc": self.out})
        return L


MODELS = {"resnet8": ResNet8, "dscnn": DSCNN, "fcae": FCAutoEncoder}
