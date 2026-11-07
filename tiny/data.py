"""Datasets of the MLPerf Tiny benchmarks.

load() returns a dict:
  train, val, test   (x, y) tensors on the device. CIFAR-10: uint8 images NCHW. Speech Commands: train holds raw
                     int16 audio (features are computed per batch, after augmentation); val and test hold MFCCs.
  augment(x, gen)    a training batch -> the float model's input
  to_float(x)        a stored val/test batch -> the float model's input
  calib              inputs for activation calibration (stored form, like val/test)
  signed             False: the integer network reads 8-bit pixels; True: signed features (MFCCs, log-mels)
  task               "classify" (accuracy) or "ae" (reconstruction; anomaly AUC per machine on the test set)
  opt, batch         optimizer ("sgd" or "adam") and batch size used for training

Images are not mean/std normalized: the first layer sees raw pixels, so the integer program's input is the
8-bit pixel itself and zero padding means black, as on a camera-fed device.
"""
import math
import pathlib
import pickle
import random
import wave

import numpy as np
import torch

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
CLASSES = {"cifar10": 10, "speech_commands": 12, "toyadmos": 0}


def _cifar10():
    root = DATA / "cifar10" / "cifar-10-batches-py"

    def load(names):
        xs, ys = [], []
        for n in names:
            with open(root / n, "rb") as f:
                d = pickle.load(f, encoding="bytes")
            xs.append(d[b"data"].reshape(-1, 3, 32, 32))
            ys += d[b"labels"]
        return np.concatenate(xs), np.array(ys)

    return load([f"data_batch_{i}" for i in range(1, 6)]), load(["test_batch"])


def load(name, device, val=5000):
    """See the module docstring. CIFAR-10: the last `val` training images are held out for choices such as the
    activation calibration; the test set is only used for the reported numbers."""
    if name == "speech_commands":
        return _speech_commands(device)
    if name == "toyadmos":
        return _toyadmos(device)
    if name != "cifar10":
        raise ValueError(name)
    (xtr, ytr), (xte, yte) = _cifar10()
    t = lambda x, y: (torch.tensor(x, device=device), torch.tensor(y, dtype=torch.long, device=device))
    n = len(xtr) - val
    d = {"train": t(xtr[:n], ytr[:n]), "val": t(xtr[n:], ytr[n:]), "test": t(xte, yte)}
    return {**d, "augment": lambda x, gen: augment(x, gen).float() / 255, "to_float": lambda x: x.float() / 255,
            "calib": d["train"][0][:5000], "signed": False, "task": "classify", "opt": "sgd", "batch": 128}


def augment(x, gen):
    """CIFAR augmentation on a uint8 GPU batch: random crop with 4-pixel zero padding + horizontal flip."""
    b = x.shape[0]
    xp = torch.nn.functional.pad(x, (4, 4, 4, 4))
    oy = torch.randint(0, 9, (b,), device=x.device, generator=gen)
    ox = torch.randint(0, 9, (b,), device=x.device, generator=gen)
    ar = torch.arange(32, device=x.device)
    iy = (oy[:, None] + ar)[:, :, None]                       # (B, 32, 1)
    ix = (ox[:, None] + ar)[:, None, :]                       # (B, 1, 32)
    flip = torch.rand(b, device=x.device, generator=gen) < 0.5
    ix = torch.where(flip[:, None, None], ix.flip(-1), ix)
    out = xp.permute(0, 2, 3, 1)[torch.arange(b, device=x.device)[:, None, None], iy, ix]   # (B, 32, 32, C)
    return out.permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------- Speech Commands v2 (keyword spotting)

KWS_WORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
KWS_LABELS = KWS_WORDS + ["_silence_", "_unknown_"]
SR, CLIP = 16000, 16000
FRAME, HOP, NFFT, NMEL, NMFCC, FMIN, FMAX = 480, 320, 512, 40, 10, 20.0, 4000.0


def _read_wav(path):
    with wave.open(str(path)) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    out = np.zeros(CLIP, dtype=np.int16)
    out[:min(len(a), CLIP)] = a[:CLIP]
    return out


def _mel_matrix(device):
    hz2mel = lambda f: 1127.0 * math.log1p(f / 700.0)
    edges = torch.linspace(hz2mel(FMIN), hz2mel(FMAX), NMEL + 2, dtype=torch.float64)
    freqs = torch.linspace(0, SR / 2, NFFT // 2 + 1, dtype=torch.float64)
    mel = 1127.0 * torch.log1p(freqs / 700.0)
    lo, ce, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    m = torch.clamp(torch.minimum((mel - lo) / (ce - lo), (hi - mel) / (hi - ce)), min=0)
    m[:, 0] = 0                                                            # no DC, as in TensorFlow
    return m.T.float().to(device)                                          # (bins, mels)


def _dct_matrix(device):
    n = torch.arange(NMEL, dtype=torch.float64)
    k = torch.arange(NMFCC, dtype=torch.float64)[:, None]
    d = torch.cos(math.pi / NMEL * (n + 0.5) * k) * math.sqrt(2.0 / NMEL)
    d[0] /= math.sqrt(2.0)                                                 # orthonormal DCT-II
    return d.T.float().to(device)                                          # (mels, mfcc)


_FEAT = {}


def mfcc(audio):
    """int16 or float audio (N, 16000) -> MFCC (N, 1, 49, 10): 30 ms Hann frames every 20 ms, |FFT|, 40 mel bands
    from 20 Hz to 4 kHz, log, DCT-II, first 10 coefficients (the MLPerf Tiny keyword-spotting front end)."""
    dev = audio.device
    if dev not in _FEAT:
        _FEAT[dev] = (torch.hann_window(FRAME, periodic=True, device=dev), _mel_matrix(dev), _dct_matrix(dev))
    win, mel, dct = _FEAT[dev]
    x = audio.float() / 32768.0 if audio.dtype == torch.int16 else audio.float()
    frames = x.unfold(1, FRAME, HOP) * win                                 # (N, 49, 480)
    spec = torch.fft.rfft(frames, n=NFFT).abs()                            # (N, 49, 257)
    logmel = torch.log(spec @ mel + 1e-6)
    return (logmel @ dct).unsqueeze(1)


def _speech_commands(device):
    root = DATA / "speech_commands"
    cache = root / "cache_v1.pt"
    if not cache.exists():
        v2 = root / "v2"
        listed = lambda f: set((v2 / f).read_text().split())
        val_set, test_set = listed("validation_list.txt"), listed("testing_list.txt")
        rng = random.Random(0)
        split = {"train": ([], []), "val": ([], [])}
        unknown = {"train": [], "val": []}
        for d in sorted(p for p in v2.iterdir() if p.is_dir() and not p.name.startswith("_")):
            for f in sorted(d.glob("*.wav")):
                rel = f"{d.name}/{f.name}"
                part = "val" if rel in val_set else None if rel in test_set else "train"
                if part is None:
                    continue
                if d.name in KWS_WORDS:
                    split[part][0].append(_read_wav(f))
                    split[part][1].append(KWS_LABELS.index(d.name))
                else:
                    unknown[part].append(f)
        noise = [_read_wav_full(f) for f in sorted((v2 / "_background_noise_").glob("*.wav"))]
        for part, (xs, ys) in split.items():
            n_words = len(xs)
            # 10% unknown words and 10% silence, as in the Speech Commands reference pipeline
            for f in rng.sample(unknown[part], n_words // 8):
                xs.append(_read_wav(f))
                ys.append(KWS_LABELS.index("_unknown_"))
            for _ in range(n_words // 8):
                nz = noise[rng.randrange(len(noise))]
                o = rng.randrange(len(nz) - CLIP)
                xs.append((nz[o:o + CLIP] * rng.uniform(0, 1)).astype(np.int16))
                ys.append(KWS_LABELS.index("_silence_"))
        tx, ty = [], []
        for lab in KWS_LABELS:
            for f in sorted((root / "test" / lab).glob("*.wav")):
                tx.append(_read_wav(f))
                ty.append(KWS_LABELS.index(lab))
        T = lambda a: torch.tensor(np.stack(a))
        torch.save({"train": (T(split["train"][0]), torch.tensor(split["train"][1])),
                    "val": (T(split["val"][0]), torch.tensor(split["val"][1])),
                    "test": (T(tx), torch.tensor(ty)),
                    "noise": torch.tensor(np.concatenate(noise))}, cache)
    c = torch.load(cache)
    g = lambda t: (t[0].to(device), t[1].long().to(device))
    train, val, test = g(c["train"]), g(c["val"]), g(c["test"])
    noise = c["noise"].to(device).float() / 32768.0
    feats = lambda a: torch.cat([mfcc(a[i:i + 2000]) for i in range(0, len(a), 2000)])

    def augment_audio(x, gen):
        """Random time shift of up to 100 ms and background noise (p = 0.8, volume up to 0.1)."""
        b = x.shape[0]
        a = x.float() / 32768.0
        shift = torch.randint(-1600, 1601, (b,), device=x.device, generator=gen)
        idx = torch.arange(CLIP, device=x.device)[None] - shift[:, None]
        a = torch.where((idx >= 0) & (idx < CLIP), a.gather(1, idx.clamp(0, CLIP - 1)), torch.zeros_like(a))
        off = torch.randint(0, len(noise) - CLIP, (b,), device=x.device, generator=gen)
        nz = noise[off[:, None] + torch.arange(CLIP, device=x.device)[None]]
        vol = torch.rand(b, 1, device=x.device, generator=gen) * 0.1
        vol = vol * (torch.rand(b, 1, device=x.device, generator=gen) < 0.8)
        return mfcc(torch.clamp(a + vol * nz, -1, 1))

    return {"train": train, "val": (feats(val[0]), val[1]), "test": (feats(test[0]), test[1]),
            "augment": augment_audio, "to_float": lambda x: x, "calib": feats(train[0][:5000]), "signed": True,
            "task": "classify", "opt": "sgd", "batch": 128}


def _read_wav_full(path):
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).copy()


# ---------------------------------------------------------------- ToyADMOS (DCASE 2020 task 2, ToyCar)

AD_NFFT, AD_HOP, AD_MELS, AD_FRAMES = 1024, 512, 128, 5


def logmel_vectors(audio):
    """float audio (N, samples) at 16 kHz -> (N, T, 640): 10 log10 of the mel power spectrum (1024-point frames
    every 512 samples, centered, 128 bands up to 8 kHz), 5 consecutive frames per vector (the DCASE baseline)."""
    dev = audio.device
    key = ("ad", dev)
    if key not in _FEAT:
        hz2mel = lambda f: 1127.0 * math.log1p(f / 700.0)
        edges = torch.linspace(0.0, hz2mel(SR / 2), AD_MELS + 2, dtype=torch.float64)
        mel = 1127.0 * torch.log1p(torch.linspace(0, SR / 2, AD_NFFT // 2 + 1, dtype=torch.float64) / 700.0)
        lo, ce, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
        m = torch.clamp(torch.minimum((mel - lo) / (ce - lo), (hi - mel) / (hi - ce)), min=0)
        _FEAT[key] = (torch.hann_window(AD_NFFT, periodic=True, device=dev), m.T.float().to(dev))
    win, m = _FEAT[key]
    spec = torch.stft(audio.float(), AD_NFFT, AD_HOP, window=win, center=True, pad_mode="reflect",
                      return_complex=True).abs() ** 2                       # (N, 513, T)
    lm = 10.0 * torch.log10(spec.transpose(1, 2) @ m + 1e-12)               # (N, T, 128)
    t = lm.shape[1] - AD_FRAMES + 1
    return torch.cat([lm[:, i:i + t] for i in range(AD_FRAMES)], dim=2)     # (N, T-4, 640)


def _toyadmos(device):
    root = DATA / "toyadmos" / "ToyCar"
    cache = DATA / "toyadmos" / "cache_v1.pt"
    if not cache.exists():
        def clips(part):
            fs = sorted((root / part).glob("*.wav"))
            meta = [(f.name.startswith("anomaly"), int(f.name.split("_id_")[1][:2])) for f in fs]
            return fs, meta

        def vectors(fs):
            out = []
            for i in range(0, len(fs), 64):
                a = torch.tensor(np.stack([_read_wav_full(f) for f in fs[i:i + 64]])).float().to(device) / 32768.0
                out.append(logmel_vectors(a).half().cpu())
            return torch.cat(out)                                           # (clips, T, 640)

        tr, _ = clips("train")
        te, te_meta = clips("test")
        rng = random.Random(0)
        idx = list(range(len(tr)))
        rng.shuffle(idx)
        nval = len(tr) // 10                                                # 10% of the normal clips: validation
        torch.save({"train": vectors([tr[i] for i in sorted(idx[nval:])]),
                    "val": vectors([tr[i] for i in sorted(idx[:nval])]),
                    "test": vectors(te), "test_anomaly": torch.tensor([m[0] for m in te_meta]),
                    "test_machine": torch.tensor([m[1] for m in te_meta])}, cache)
    c = torch.load(cache)
    t = c["train"].shape[1]
    as_input = lambda v: v.reshape(-1, 640, 1, 1)
    # training uses every other vector of each clip (they overlap by 4 of 5 frames); val and test use all
    train = as_input(c["train"][:, ::2]).to(device)
    val = as_input(c["val"]).to(device)
    test = as_input(c["test"]).to(device)
    nclip = c["test"].shape[0]
    info = {"clip": torch.arange(nclip).repeat_interleave(t), "anomaly": c["test_anomaly"],
            "machine": c["test_machine"]}
    zeros = lambda x: torch.zeros(len(x), dtype=torch.long, device=device)
    return {"train": (train, zeros(train)), "val": (val, zeros(val)),
            "test": (test, c["test_anomaly"].repeat_interleave(t).to(device)), "test_info": info,
            "augment": lambda x, gen: x.float(), "to_float": lambda x: x.float(), "calib": train[:50000],
            "signed": True, "task": "ae", "opt": "adam", "batch": 512,
            "qat_scale": 1 / 3}                       # an epoch is 612k vectors: a third of the QAT epochs
