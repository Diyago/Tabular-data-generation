---
title: "TabGAN: Generate Synthetic Tabular Data with GANs, Diffusion Models & LLMs in 3 Lines of Python"
thumbnail: /blog/assets/tabgan/thumbnail.png
authors:
  - user: InsafQ
date: 2026-03-29
tags:
  - synthetic-data
  - tabular
  - gan
  - diffusion
  - privacy
  - open-source
---

# TabGAN: Generate Synthetic Tabular Data with GANs, Diffusion & LLMs — in 3 Lines of Python

**TL;DR:** [TabGAN](https://github.com/Diyago/Tabular-data-generation) lets you generate high-quality synthetic tabular data using GANs, Forest Diffusion, or LLMs — with built-in quality reports, privacy metrics, and now **AutoSynth** (auto-picks the best generator) and **one-click synthesis for any HuggingFace dataset**.

## The Problem

You have tabular data that's too sensitive to share, too small to train on, or too imbalanced to model well. You need synthetic data that:

- **Preserves statistical properties** of the original
- **Doesn't memorize** individual records (privacy!)
- **Works out of the box** without ML PhD-level tuning

## The Solution: TabGAN

```bash
pip install tabgan
```

### 3 Lines to Synthetic Data

```python
from tabgan import GANGenerator
import pandas as pd

df = pd.read_csv("your_data.csv")
gen = GANGenerator(gen_x_times=1.1, cat_cols=["gender", "city"])
synthetic, _ = gen.generate_data_pipe(df, None, df, only_generated_data=True)
```

That's it. `synthetic` is a DataFrame with realistic rows that never existed in the original data.

## What Makes TabGAN Different?

### 🔄 One API, Multiple Generators

Switch between state-of-the-art methods with a single parameter change:

| Generator | Best For | Speed |
|-----------|----------|-------|
| **CTGAN** (GAN) | General purpose, mixed types | Fast |
| **Forest Diffusion** | Tree-friendly structured data | Medium |
| **LLM** (GReaT) | Text-rich, semantic dependencies | Slow |
| **Random Baseline** | Quick benchmarking | Instant |

```python
from tabgan import GANGenerator, ForestDiffusionGenerator, LLMGenerator

# Just swap the class — same API!
gen = ForestDiffusionGenerator(gen_x_times=1.0, cat_cols=["category"])
synthetic, _ = gen.generate_data_pipe(df, target, df, only_generated_data=True)
```

### 🏆 NEW: AutoSynth — Let the Library Choose

Don't know which generator works best for your data? **AutoSynth** runs all of them and picks the winner:

```python
from tabgan import AutoSynth

result = AutoSynth(df, target_col="label").run()

print(result.report)
#   Generator          Status  Score  Quality  Privacy  Rows  Time (s)
# 0 GAN (CTGAN)        OK      0.847  0.891    0.743    165   12.3
# 1 Forest Diffusion   OK      0.812  0.834    0.761    165   45.1
# 2 Random Baseline    OK      0.654  0.621    0.732    165   0.1

best_synthetic = result.best_data  # Best generator's output
print(f"Winner: {result.best_name}")  # "GAN (CTGAN)"
```

AutoSynth scores each generator on a weighted combination of **quality** (distribution fidelity, ML utility) and **privacy** (distance to closest record, membership inference risk).

### 🤗 NEW: One-Click Synthesis for Any HuggingFace Dataset

Synthesize any tabular dataset from the Hub — and push the result back:

```python
from tabgan import synthesize_hf_dataset

# Load → Generate → Evaluate in one call
result = synthesize_hf_dataset(
    "scikit-learn/iris",
    target_col="target",
)

# Push synthetic version to your HF account
result = synthesize_hf_dataset(
    "scikit-learn/iris",
    target_col="target",
    push_to_hub=True,
    hub_repo_id="your-username/iris-synthetic",
)
```

### 📊 Built-in Quality & Privacy Reports

Every generation can be evaluated automatically:

**Quality Report** — PSI (distribution divergence), correlation comparison, ML utility (train-on-synthetic, test-on-real):

```python
from tabgan import QualityReport

report = QualityReport(original_df, synthetic_df, cat_cols=["gender"], target_col="label")
report.compute()
report.to_html("quality_report.html")  # Self-contained HTML with plots
```

**Privacy Metrics** — Distance to Closest Record, Nearest Neighbor Distance Ratio, Membership Inference Risk:

```python
from tabgan import PrivacyMetrics

pm = PrivacyMetrics(original_df, synthetic_df, cat_cols=["gender"])
summary = pm.summary()
print(f"Privacy score: {summary['overall_privacy_score']}")  # 0 = leaked, 1 = private
```

### 🔧 Business Constraints

Enforce domain rules on generated data:

```python
from tabgan import GANGenerator, RangeConstraint, FormulaConstraint

gen = GANGenerator(
    gen_x_times=1.5,
    cat_cols=["department"],
    constraints=[
        RangeConstraint("age", min_val=18, max_val=65),
        RangeConstraint("salary", min_val=0),
        FormulaConstraint("end_date > start_date"),
    ],
)
```

### 🔌 sklearn Pipeline Integration

Drop synthetic augmentation into any ML pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from tabgan import TabGANTransformer

pipe = Pipeline([
    ("augment", TabGANTransformer(gen_x_times=2.0, cat_cols=["gender"])),
    ("model", RandomForestClassifier()),
])
pipe.fit(X_train, y_train)
```

## Try It Now

**Interactive Demo:** [insafq-tabgan.hf.space](https://insafq-tabgan.hf.space)

**Install:**
```bash
pip install tabgan
```

**GitHub:** [github.com/Diyago/Tabular-data-generation](https://github.com/Diyago/Tabular-data-generation)

**PyPI:** [pypi.org/project/tabgan](https://pypi.org/project/tabgan/)

## Benchmarks

### Quality (Normalized ROC AUC)

| Dataset | CTGAN | Forest Diffusion | Random |
|---------|-------|-------------------|--------|
| Credit | 0.752 | 0.781 | 0.501 |
| Adult Census | 0.689 | 0.712 | 0.523 |
| Telecom | 0.814 | 0.799 | 0.548 |

*Higher is better.*

### Speed (generation time, 1000 rows, 8 features)

| Generator | Time | Notes |
|-----------|------|-------|
| **Random Baseline** | ~0.1s | Instant — just resampling |
| **CTGAN (GAN)** | ~1–10s | Fast, depends on epochs |
| **Forest Diffusion** | ~30–120s | High quality, but slower |
| **LLM (GReaT)** | ~5–30min | Best for text columns, GPU recommended |

Every `generate_data_pipe()` call now records per-step timing in `generator.last_timing_`:

```python
gen = GANGenerator(gen_x_times=1.1)
synthetic, _ = gen.generate_data_pipe(train, target, test)
print(gen.last_timing_)
# {'preprocess': 0.001, 'generation': 2.3, 'postprocess': 0.01,
#  'adversarial_filtering': 0.15, 'total': 2.46}
```

*Full benchmarks in the [README](https://github.com/Diyago/Tabular-data-generation).*

## What's Next

- **Public Leaderboard** for synthetic tabular data generators
- **Differential Privacy** guarantees (DP-SGD)
- **Natural language generation** — "Generate 1000 patients aged 20-40"

---

*TabGAN is Apache 2.0 licensed. Contributions welcome!*

*Star the repo if you find it useful:* ⭐ [github.com/Diyago/Tabular-data-generation](https://github.com/Diyago/Tabular-data-generation)
