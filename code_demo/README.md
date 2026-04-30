# Code demo — BaVarIA / BASE hierarchy

A small, self-contained sample so a reviewer can run the attacks and reproduce
two ROC figures from the paper. One testbed only (Location/MLP3, 5,010
samples), 5 antithetic pairs of shadow models. The full 12-testbed × 128-pair
corpus used in the paper is too large to share via this anonymous repository;
we ship the minimum the reviewer needs to run the demo end-to-end and verify
that the cleaned-up numpy methods agree with the paper's original pipeline
on the same data.

## Provenance

`methods.py` is a numpy rewrite of the same likelihood formulae we used to
produce the paper; the original implementation is vectorised and uses GPU
for RMIA. `data/reference_scores_K8.npz` contains scores from the original
code on this exact bundle, and the last cell of the notebook checks numerical
agreement so this isn't taken on trust. All eight (method × setting)
combinations match within float32 round-off.

## Layout

```
code_demo/
├── README.md
├── requirements.txt
├── data/                            (~420 KB total)
│   ├── shadow_logits.npy          # (5010, 10) float32
│   ├── shadow_mask.npy            # (5010, 10) bool
│   ├── target_logits.npy          # (5010,)    float32
│   ├── target_mask.npy            # (5010,)    bool
│   ├── reference_scores_K8.npz    # scores from the original pipeline
│   └── meta.json
├── methods.py                     # LiRA, BASE1–4, RMIA, BaVarIA-n, BaVarIA-t
├── utils.py                       # bundle loader, sampler, metrics
└── demo.ipynb                     # produces the two figures + verification
```

The 10 shadow columns are 5 antithetic pairs: columns `2i` and `2i+1` have
complementary membership masks. Holding out one pair as the audit target
leaves 4 pairs of shadow models, so `K = 8` is the largest setting the
bundle supports.

## Running

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter notebook demo.ipynb
```

About a minute on a laptop CPU. No GPU needed.

## What you should see

- **Figure 1 — BaVarIA at K = 8**, online and offline (1×2 log–log ROC).
  LiRA, BASE, BaVarIA-*n*, BaVarIA-*t*.
- **Figure 2 — BASE hierarchy at K = 4 and K = 8**, online (1×2 log–log).
  BASE1, BASE2, BASE3, BASE4 (= LiRA), RMIA. The paper additionally shows
  K = 64 and K = 254; those need a larger pair pool than the 5 shipped here.
- **Verification**: a final cell prints `OK` for each of the 8 method ×
  setting combinations, confirming the numpy methods match the original
  pipeline element-wise on a fixed `(target_pair, shadow_idx)` configuration.

## What is left out

To keep the archive small and the demo focused: ELSA, the full 12-testbed ×
18-method benchmark, the figure-generation scripts, and the DP-SGD pipeline.
The full project is held in the (deanonymised) project repository and can
be made available on request.

## Method references

- **LiRA** — Carlini, Chien, Nasr, Song, Terzis, Tramèr. *Membership Inference
  Attacks From First Principles.* S&P 2022.
- **RMIA** — Zarifzadeh, Liu, Shokri. *Low-Cost High-Power Membership
  Inference by Boosting Relativity.* 2023.
- **BASE / BaVarIA** — this paper.
