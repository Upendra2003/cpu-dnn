import gzip
import pathlib

import numpy as np
import torch

DATA_ROOT = pathlib.Path(__file__).resolve().parents[1] / "data"
# name -> (folder under data/, mean, std of the training pixels in [0, 1])
DATASETS = {
    "mnist": ("mnist", 0.1307, 0.3081),
    "fashion": ("fashion", 0.2860, 0.3530),
}


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_dataset(name, device):
    """Returns ((x_train, y_train), (x_test, y_test)) as normalised tensors on `device`."""
    folder, mean, std = DATASETS[name]
    out = []
    for split in ("train", "t10k"):
        x = _read_idx(DATA_ROOT / folder / f"{split}-images-idx3-ubyte.gz")
        y = _read_idx(DATA_ROOT / folder / f"{split}-labels-idx1-ubyte.gz")
        x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(1) / 255.0
        out.append(((x - mean) / std, torch.tensor(y, dtype=torch.long, device=device)))
    return out
