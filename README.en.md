# CalibGuard (CG)

English | [中文](README.md)

**A low-cost file security auditing system** — a small local model gates the traffic, and a cloud LLM reviews only what needs it.

Built for casual developers, CS students, and AI beginners: drop in a suspicious file
(script / code / text / archive) and get one of four verdicts:
**allow / suspicious / block / manual review**.

---

## How it works

```
                     ┌─────────── Cloud LLM review (optional, gray zone only) ───────────┐
                     ▼                                                                  │
file → preprocess → feature extraction → CodeBERT sliding window → Platt calibration → three-bucket gate ─┐
                                                                                                          │
                          p < lower  → allow     (no network)
                          p > upper  → block     (no network)
                          in between → cloud review (falls back to p>0.5; can be disabled)
```

Three key design choices:

1. **Probability calibration** — the fine-tuned model's raw logit is mapped by Platt scaling into an
   interpretable "malicious probability". Threshold decisions become far more reliable than raw logits
   (calibration costs ~1.4pt of F1 but buys a great deal of interpretability).
2. **Three-bucket gate** — the vast majority of files (clearly benign or clearly malicious) are decided
   with no network call at all. Only the gray zone is paid for, balancing cost against recall.
3. **Full-text sliding window** — inference covers the whole document with 512-token windows and a
   256-token stride, aggregating `max(window logit)`, so it is not limited by a single 512-token truncation.

---

## Results (test split, 25,049 samples, uniform `p > 0.5` decision rule)

| Backbone | Raw F1 | Calibrated F1 | Calibrated P | Calibrated R | Training time |
|---|---|---|---|---|---|
| **CodeBERT (batch 16, recommended)** | 0.8734 | **0.8600** | 0.8740 | 0.8464 | 2.43h |
| CodeBERT (batch 8) | 0.8691 | 0.8672 | 0.8200 | 0.9201 | 2.54h |
| Qwen2.5-Coder-0.5B-Instruct | 0.8685 | 0.8396 | 0.8601 | 0.8202 | 5.69h |
| Qwen3-0.6B | 0.8572 | 0.8295 | 0.8440 | 0.8156 | 5.7h |
| Qwen3-CoderSmall | 0.8582 | 0.8276 | 0.8620 | 0.7958 | 7.8h |

Baselines: rule-based F1 0.5025; legacy version v0.2.2 raw F1 0.5953.

**With cloud LLM review on the gray zone**: F1 0.8600 → **0.8689**, precision 0.8739 → **0.9017**.

> Why CodeBERT: it runs at **22.4 files/s** (3–4.3× the 0.5B-class Qwen models), trains 2.3–3.2× faster,
> and is 4–5× smaller — the best overall trade-off, hence the default deployment backbone.

> Full evaluation details (ablation matrix, per-domain recall, three-bucket threshold calibration,
> error analysis) are in the [**model performance comparison report**](CG_Bench/模型性能对比分析报告.md) (Chinese).

---

## Repository layout

```
CG/
├── CG_GUI/                 Desktop shell: launcher + backend + single-page frontend (what users run)
│   ├── launcher.py         Launcher (stdlib only; bootstraps the environment on first run)
│   ├── server.py           FastAPI backend
│   ├── audit_service.py    Audit job wrapper (single concurrency)
│   ├── calibguard/         Engine copy
│   ├── web/index.html      Single-page frontend (self-contained, works offline)
│   ├── samples/            Demo samples
│   ├── tests/              End-to-end acceptance tests
│   ├── tools/              Verification scripts
│   ├── README.txt          End-user manual (Chinese)
│   └── requirements-gui.txt
└── CG_Bench/               Benchmark harness: training / evaluation / ablations (produces the model)
    ├── harness.py          Seven-step pipeline orchestration (HPO→train→eval→calibrate→…)
    ├── train/              Training, evaluation, calibration, HPO engines
    ├── adapters/           LLM / hybrid / rules / zero-shot comparison adapters
    ├── calibguard/         Engine copy
    ├── tests/
    ├── requirements.txt
    └── setup.sh / setup.bat
```

> Each subproject ships its own copy of the `calibguard/` engine so it can run and be packaged independently.

---

## Quick start

### Option 1: Download and run (recommended, no Python required)

1. Go to this repository's [**Releases**](../../releases) page and download `CalibGuard-GUI.zip`
2. Extract it to any directory (extract **the whole folder** — do not keep only the exe)
3. Double-click `CalibGuard.exe` — the first run prepares the environment automatically (about 2–10 minutes, needs internet)
4. When it is ready, your browser opens automatically; drag files in to audit them

See `README.txt` inside the archive for details. Model weights are bundled, so **no training is needed**.

### Option 2: Run from source (developers)

**Desktop shell**:

```bash
cd CG_GUI
python -m venv .venv
.venv/Scripts/pip install -r requirements-gui.txt      # on Linux/macOS use .venv/bin/pip
.venv/Scripts/python launcher.py
```

> Requires a `models/` directory (model + calibration parameters). See "Models and data" below.

**Benchmark harness (training / evaluation)**:

```bash
cd CG_Bench
bash setup.sh                 # Linux; use setup.bat on Windows
bash train-bench.sh --dry-run # preview the command sequence first
bash train-bench.sh           # run experiments serially in experiments.yaml order
```

---

## Configuration

The desktop shell's settings panel (top-right "Settings") lets you configure:

| Setting | Description |
|---|---|
| Use cloud LLM review | Master switch. When off, the gray zone is decided by the local model at 50% |
| API Key / Base URL / Model name | Any OpenAI-compatible endpoint (DeepSeek by default) |
| Content to submit | Distilled evidence / raw file text |
| Submission mode | Truncate (first N chars) / sliding window (multi-chunk, take the most severe) / send whole |
| Gray zone range | 0.15 ~ 0.85 by default |

Settings are stored locally in `gui-settings.json` and are **never committed**.

The harness reads its key from an environment variable or a project-root `.env` (`DEEPSEEK_API_KEY`), also **never committed**.

---

## Models and data

To keep the repository lightweight, the following are **not distributed with the repository**:

| Item | Notes |
|---|---|
| Training / evaluation corpus | ~26 GB, contains third-party package content; not suitable for public distribution |
| Training artifacts `results/` | ~8 GB of intermediate output |
| Model weights | Bundled in `CalibGuard-GUI.zip` on the Releases page — ready to use |

To reproduce training yourself: prepare the corpus, then follow `CG_Bench/experiments.yaml`.
The data splitting, HPO grid, and validation flow all live in `harness.py` and `train/`.

---

## Known limitations

- The corpus is dominated by the Python ecosystem (PyPI / npm); recall drops for other language ecosystems.
- Cloud review is **optional**: without an API key, gray-zone files are routed to manual review, and all other verdicts are unaffected.
- It runs without a discrete GPU (CPU build is selected automatically) but inference is slower.
- Older GPUs with compute capability below 7.5 (e.g. the GTX 10 series) automatically fall back to the CPU build — recent acceleration libraries no longer support their compute capability.

---

## License

See [LICENSE](LICENSE).
