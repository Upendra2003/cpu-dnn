# RoBA approximate multiplication for a DNN on a multiplierless CPU

Phase 1 (GPU): train an MNIST DNN and evaluate it with the approximate multipliers from
Rajiv & Kedia, DATE'26, before porting to Ibex RV32I + LOD.

```
roba/multipliers.py   exact, Mitchell, DRUM6, RoBA-Dn1/UpDn/Dn2/UpDnDn/(UpDn)2 (PyTorch, bit-exact)
roba/matmul.py        approximate linear / conv2d (RoBA & DRUM as a few exact matmuls)
roba/reference.py     scalar bitwise reference used by the tests
roba/nn.py            approximate layers, quantization (float / trunc / int8 / int16), MLP, LeNet-5
roba/cost.py          cycle & energy ESTIMATES from the paper's Fig. 1/4 (until Ibex measurement)
csrc/roba.{c,h}       C library for Ibex (LOD via custom instruction or __builtin_clz)
tests/                PyTorch == Python reference == C, matmul/conv == elementwise
experiments/run.py    stages: errors, train, eval, finetune
```

```
# MNIST (not tracked in git)
mkdir -p data/mnist && for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -sfL -o data/mnist/$f.gz https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz; done

python3 -m pytest -q tests
python3 experiments/run.py --stage all     # results/*.csv, log in results/run.log
python3 experiments/build_site.py          # index.html (interactive results page) from results/
```

Notes
- "Mitchell" here is the classic algorithm (with carry, max error 11.1%). The paper's Mitchell
  (following Babic et al.) gives the same products as RoBA-Dn1 (identical SNR in Table I).
- `trunc` domain = floats cast to int without scaling; it collapses to chance accuracy and is
  kept to show why scaling (fixed point) is required.
