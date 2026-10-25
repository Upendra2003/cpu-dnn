import gzip
import pathlib

import numpy as np
import torch

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "mnist"
MEAN, STD = 0.1307, 0.3081


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_mnist(device):
    """Returns ((x_train, y_train), (x_test, y_test)) as normalised tensors on `device`."""
    out = []
    for split in ("train", "t10k"):
        x = _read_idx(DATA_DIR / f"{split}-images-idx3-ubyte.gz")
        y = _read_idx(DATA_DIR / f"{split}-labels-idx1-ubyte.gz")
        x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(1) / 255.0
        out.append(((x - MEAN) / STD, torch.tensor(y, dtype=torch.long, device=device)))
    return out
