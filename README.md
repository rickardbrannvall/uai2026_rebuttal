# UAI 2026 — supplementary materials and code demo

This anonymous repository accompanies the rebuttal for UAI 2026 Submission 642.
It contains the supplementary PDF reports referenced in our responses and a
small, self-contained code demo.

## Contents

```
.
├── README.md                  this file
├── summary32.pdf              short, curated overview of the headline results
├── results/
│   ├── base32.pdf             BASE-hierarchy report (BASE1, BASE2*, BASE3, BASE4)
│   ├── bavaria32.pdf            LiRA vs BaVarIA report (online + offline)
│   └── dpsgd.pdf              DP-SGD evaluation
└── code_demo/                 runnable methods + minimal data sample
```

### PDF reports

- **`summary32.pdf`** — selected figures and tables from the two longer
  reports, presented two testbeds at a time (CIFAR-100/WRN and Texas/MLP3
  throughout). Includes a weighted-concordance table that quantifies the
  BASE hierarchy ordering. Read this first.
- **`results/base32.pdf`** — the full BASE-hierarchy comparison across
  12 testbeds, seven shadow-model budgets ($K \in \{4, 8, 16, 32, 64, 128, 254\}$),
  and 32 leave-one-out replicates per configuration. Includes per-testbed
  tables and ROC curves at $K = 8$, $64$, $254$ for each method.
- **`results/bavaria32.pdf`** — the corresponding LiRA / BASE / BaVarIA-$n$ /
  BaVarIA-$t$ comparison, covering both online and offline settings.
- **`results/dpsgd.pdf`** — evaluation under DP-SGD at four noise levels
  (CIFAR-10/WRN, CIFAR-100/WRN, Location/MLP3, Purchase-100/MLP3, $K = 64$).

All numbers in the reports use **leave-one-out replicates** (every shadow
model audited as the held-out target). Means and ±1 SE bands unless noted.

### Code demo

The `code_demo/` subdirectory contains a numpy rewrite of the attack scoring
functions (`methods.py`) plus a Jupyter notebook (`demo.ipynb`) that
reproduces the headline ROC figures. To keep the archive small enough for
anonymous sharing we ship a **minimal dataset** — Location/MLP3 with
5 antithetic pairs of shadow models (5,010 samples × 10 columns, ~420 KB
in total). This is enough for the reviewer to run the demo end-to-end and
verify, via the notebook's final cell, that the methods agree element-wise
with scores produced by the paper's original pipeline on the same bundle.

The full 12-testbed × 128-pair shadow-model corpus used to produce the PDF
reports is too large to ship here; it is held in the project repository
and can be made available on request after deanonymisation.

See `code_demo/README.md` for run instructions.
