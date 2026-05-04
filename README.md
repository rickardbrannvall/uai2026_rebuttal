# UAI 2026 Submission 642 — supplementary figures

```
.
├── summary32.pdf
├── results/
│   ├── base32.pdf
│   ├── bavaria32.pdf
│   └── dpsgd.pdf
└── code_demo/
```

| Path | Contents |
|---|---|
| `summary32.pdf` | Curated headline figures and concordance table for the BASE hierarchy and BaVarIA. Read first. |
| `results/base32.pdf` | BASE hierarchy: per-testbed tables and ROC curves across $K \in \{4, 8, 16, 32, 64, 128, 254\}$. |
| `results/bavaria32.pdf` | LiRA vs. BaVarIA-$n$ vs. BaVarIA-$t$: per-testbed tables and ROC curves, online and offline. |
| `results/dpsgd.pdf` | Membership inference under DP-SGD ($\sigma \in \{0, 0.01, 0.1, 1.0\}$, four testbeds, $K = 64$). |
| `code_demo/` | Self-contained numpy rewrite of the attack scoring + minimal Location/MLP3 data + Jupyter notebook reproducing two ROC figures end-to-end. |
