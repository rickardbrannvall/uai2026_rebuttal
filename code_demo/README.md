# Code demo — UAI 2026 Submission 642

Minimal Location/MLP3 sample (5,010 samples, 5 antithetic shadow-pairs, $K = 8$ max). Reproduces two ROC figures and verifies element-wise agreement with the paper's original pipeline on the same bundle.

```
code_demo/
├── README.md
├── requirements.txt
├── methods.py            # LiRA, BASE1-4, RMIA, BaVarIA-n, BaVarIA-t (numpy)
├── utils.py              # bundle loader, sampler, metrics
├── demo.ipynb            # produces the two figures + verification cell
└── data/                 # ~420 KB total
    ├── shadow_logits.npy        # (5010, 10) float32
    ├── shadow_mask.npy          # (5010, 10) bool
    ├── target_logits.npy        # (5010,)    float32
    ├── target_mask.npy          # (5010,)    bool
    ├── reference_scores_K8.npz  # scores from the original pipeline
    └── meta.json
```

| Path | Contents |
|---|---|
| `methods.py` | Numpy rewrite of LiRA, BASE1-4, RMIA, BaVarIA-$n$, BaVarIA-$t$. |
| `utils.py` | Bundle loader, antithetic-pair sampler, ROC / AUC / TPR@FPR metrics. |
| `demo.ipynb` | End-to-end notebook: loads the bundle, runs all attacks, plots two ROC figures, and verifies element-wise agreement with `data/reference_scores_K8.npz`. |
| `data/shadow_logits.npy`, `shadow_mask.npy` | 10 shadow models = 5 antithetic pairs (columns $2i$ and $2i+1$ have complementary masks). |
| `data/target_logits.npy`, `target_mask.npy` | Held-out audit target. |
| `data/reference_scores_K8.npz` | Scores from the paper's original GPU pipeline on this exact bundle, for the verification cell. |
| `data/meta.json` | Bundle metadata. |
| `requirements.txt` | numpy, scipy, scikit-learn, matplotlib, jupyter. |

```bash
pip install -r requirements.txt
jupyter notebook demo.ipynb
```

About one minute on a laptop CPU. No GPU needed.
