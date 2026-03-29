# -*- coding: utf-8 -*-
"""TabGAN — Synthetic Tabular Data Generator (HuggingFace Space)."""

import io
import os
import tempfile
import traceback

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_iris, load_wine

from tabgan.sampler import GANGenerator, ForestDiffusionGenerator, OriginalGenerator
from tabgan.quality_report import QualityReport
from tabgan.privacy_metrics import PrivacyMetrics

# ---------------------------------------------------------------------------
# Demo datasets
# ---------------------------------------------------------------------------
DEMO_DATASETS = {
    "Iris (150 rows, 4 features)": "iris",
    "Wine (178 rows, 13 features)": "wine",
    "California Housing (1000 rows, 8 features)": "california",
}


def _load_demo(name: str) -> pd.DataFrame:
    if name == "iris":
        data = load_iris(as_frame=True)
        df = data.frame
        df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
        return df
    elif name == "wine":
        data = load_wine(as_frame=True)
        return data.frame
    elif name == "california":
        data = fetch_california_housing(as_frame=True)
        df = data.frame.sample(n=1000, random_state=42).reset_index(drop=True)
        return df
    raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------
GENERATORS = {
    "GAN (CTGAN)": "gan",
    "Forest Diffusion": "diffusion",
    "Random Sampling (Baseline)": "original",
}


def generate_synthetic(
    file,
    demo_choice,
    generator_name,
    gen_x_times,
    epochs,
    cat_cols_str,
    target_col,
    progress=gr.Progress(track_tqdm=True),
):
    """Main pipeline: load data → generate → quality report → privacy."""
    # --- Load data ---
    if file is not None:
        try:
            df = pd.read_csv(file.name if hasattr(file, "name") else file)
        except Exception:
            df = pd.read_excel(file.name if hasattr(file, "name") else file)
    elif demo_choice:
        key = DEMO_DATASETS.get(demo_choice)
        if not key:
            return (None, None, None, None, "Please upload a CSV or select a demo dataset.")
        df = _load_demo(key)
    else:
        return (None, None, None, None, "Please upload a CSV or select a demo dataset.")

    if len(df) < 10:
        return (None, None, None, None, "Dataset too small — need at least 10 rows.")
    if len(df.columns) < 2:
        return (None, None, None, None, "Dataset too narrow — need at least 2 columns.")

    # Limit for HF Space performance
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42).reset_index(drop=True)

    # Parse categorical columns
    cat_cols = [c.strip() for c in cat_cols_str.split(",") if c.strip()] if cat_cols_str else None
    if cat_cols:
        cat_cols = [c for c in cat_cols if c in df.columns]
        if not cat_cols:
            cat_cols = None

    # Auto-detect categorical if not specified
    if cat_cols is None:
        cat_cols = [c for c in df.columns if df[c].dtype == "object" or df[c].nunique() < 15]
    cat_cols = cat_cols or []

    # Parse target
    target_col = target_col.strip() if target_col else None
    if target_col and target_col not in df.columns:
        target_col = None

    # Separate features and target
    if target_col:
        target_series = df[[target_col]]
        train_df = df.drop(columns=[target_col])
        cat_cols_clean = [c for c in cat_cols if c != target_col]
    else:
        target_series = None
        train_df = df.copy()
        cat_cols_clean = cat_cols

    # --- Build generator ---
    gen_type = GENERATORS.get(generator_name, "gan")
    gen_params = {}

    if gen_type == "gan":
        gen_params = {"epochs": int(epochs), "batch_size": max(10, (len(train_df) // 10) // 10 * 10 or 10)}
        gen_cls = GANGenerator
    elif gen_type == "diffusion":
        gen_cls = ForestDiffusionGenerator
    else:
        gen_cls = OriginalGenerator

    try:
        generator = gen_cls(
            gen_x_times=float(gen_x_times),
            cat_cols=cat_cols_clean if cat_cols_clean else None,
            gen_params=gen_params if gen_params else None,
        )
        new_train, new_target = generator.generate_data_pipe(
            train_df,
            target_series,
            train_df,
            only_generated_data=True,
        )
    except Exception:
        return (None, None, None, None, f"Generation failed:\n```\n{traceback.format_exc()}\n```")

    # Reconstruct full synthetic df
    if target_col and new_target is not None and len(new_target.columns) > 0:
        synthetic_df = pd.concat([new_train, new_target], axis=1)
    else:
        synthetic_df = new_train

    # Align columns for comparison
    shared_cols = [c for c in df.columns if c in synthetic_df.columns]
    original_for_report = df[shared_cols]
    synthetic_for_report = synthetic_df[shared_cols]

    # --- Quality report ---
    try:
        qr = QualityReport(
            original_for_report,
            synthetic_for_report,
            cat_cols=[c for c in cat_cols if c in shared_cols],
            target_col=target_col if target_col in shared_cols else None,
        )
        qr.compute()
        quality_summary = qr.summary()

        # Save HTML
        html_path = os.path.join(tempfile.gettempdir(), "tabgan_quality_report.html")
        qr.to_html(html_path)
    except Exception:
        quality_summary = {"error": traceback.format_exc()}
        html_path = None

    # --- Privacy metrics ---
    try:
        num_cols = original_for_report.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            pm = PrivacyMetrics(
                original_for_report,
                synthetic_for_report,
                cat_cols=[c for c in cat_cols if c in shared_cols],
            )
            privacy_summary = pm.summary()
        else:
            privacy_summary = {"note": "Need >= 2 numeric columns for privacy metrics."}
    except Exception:
        privacy_summary = {"error": traceback.format_exc()}

    # --- Build visualizations ---
    fig = _build_comparison_figure(original_for_report, synthetic_for_report, cat_cols)

    # --- Format results ---
    quality_text = _format_quality(quality_summary)
    privacy_text = _format_privacy(privacy_summary)

    # --- CSV download ---
    csv_path = os.path.join(tempfile.gettempdir(), "synthetic_data.csv")
    synthetic_df.to_csv(csv_path, index=False)

    status = (
        f"Generated **{len(synthetic_df)}** synthetic rows from **{len(df)}** original rows "
        f"using **{generator_name}**."
    )

    return (
        synthetic_df.head(50),
        fig,
        quality_text + "\n\n---\n\n" + privacy_text,
        csv_path,
        status,
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def _build_comparison_figure(original: pd.DataFrame, synthetic: pd.DataFrame, cat_cols):
    num_cols = [c for c in original.columns if pd.api.types.is_numeric_dtype(original[c])]
    n_plots = min(len(num_cols), 8)
    if n_plots == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No numeric columns to plot", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    cols = min(4, n_plots)
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(num_cols[:n_plots]):
        ax = axes[i]
        ax.hist(original[col].dropna(), bins=30, alpha=0.55, label="Original", density=True, color="#4a90d9")
        ax.hist(synthetic[col].dropna(), bins=30, alpha=0.55, label="Synthetic", density=True, color="#e74c3c")
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Distribution Comparison: Original vs Synthetic", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def _format_quality(summary: dict) -> str:
    if "error" in summary:
        return f"**Quality Report Error:**\n```\n{summary['error']}\n```"

    lines = ["## Quality Report\n"]
    overall = summary.get("overall_score", "N/A")
    lines.append(f"**Overall Quality Score: {overall}**\n")

    # PSI
    psi = summary.get("psi", {})
    if psi:
        mean_psi = psi.get("mean", "N/A")
        lines.append(f"**Mean PSI:** {mean_psi:.4f}\n" if isinstance(mean_psi, float) else f"**Mean PSI:** {mean_psi}\n")
        lines.append("| Column | PSI |")
        lines.append("|--------|-----|")
        for col, val in psi.items():
            if col == "mean":
                continue
            lines.append(f"| {col} | {val:.4f} |" if isinstance(val, float) else f"| {col} | {val} |")
        lines.append("")

    # ML Utility
    ml = summary.get("ml_utility", {})
    if ml:
        lines.append(f"**ML Utility (TSTR/TRTR ratio):** {ml.get('utility_ratio', 'N/A')}")
        tstr = ml.get("tstr_auc")
        trtr = ml.get("trtr_auc")
        if tstr is not None:
            lines.append(f"- Train-Synthetic-Test-Real AUC: {tstr}")
            lines.append(f"- Train-Real-Test-Real AUC (baseline): {trtr}")
        lines.append("")

    return "\n".join(lines)


def _format_privacy(summary: dict) -> str:
    if "error" in summary:
        return f"**Privacy Metrics Error:**\n```\n{summary['error']}\n```"
    if "note" in summary:
        return f"**Privacy Metrics:** {summary['note']}"

    lines = ["## Privacy Metrics\n"]
    overall = summary.get("overall_privacy_score", "N/A")
    lines.append(f"**Overall Privacy Score: {overall}** _(0 = high risk, 1 = private)_\n")

    dcr = summary.get("dcr", {})
    if dcr:
        lines.append(f"**Distance to Closest Record (DCR):**")
        lines.append(f"- Mean: {dcr.get('mean', 'N/A'):.4f}" if isinstance(dcr.get('mean'), float) else f"- Mean: {dcr.get('mean', 'N/A')}")
        lines.append(f"- Median: {dcr.get('median', 'N/A'):.4f}" if isinstance(dcr.get('median'), float) else f"- Median: {dcr.get('median', 'N/A')}")
        lines.append(f"- 5th percentile: {dcr.get('5th_percentile', 'N/A'):.4f}" if isinstance(dcr.get('5th_percentile'), float) else f"- 5th percentile: {dcr.get('5th_percentile', 'N/A')}")
        lines.append("")

    nndr = summary.get("nndr", {})
    if nndr:
        lines.append(f"**Nearest Neighbor Distance Ratio (NNDR):**")
        lines.append(f"- Mean: {nndr.get('mean', 'N/A'):.4f}" if isinstance(nndr.get('mean'), float) else f"- Mean: {nndr.get('mean', 'N/A')}")
        lines.append(f"- Median: {nndr.get('median', 'N/A'):.4f}" if isinstance(nndr.get('median'), float) else f"- Median: {nndr.get('median', 'N/A')}")
        lines.append(f"_(Closer to 1.0 = better privacy)_")
        lines.append("")

    mi = summary.get("membership_inference", {})
    if mi:
        lines.append(f"**Membership Inference Risk:**")
        lines.append(f"- AUC: {mi.get('auc', 'N/A'):.4f}" if isinstance(mi.get('auc'), float) else f"- AUC: {mi.get('auc', 'N/A')}")
        lines.append(f"_(Closer to 0.5 = better privacy, no memorization)_")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Preview uploaded file
# ---------------------------------------------------------------------------
def preview_data(file, demo_choice):
    if file is not None:
        try:
            df = pd.read_csv(file.name if hasattr(file, "name") else file)
        except Exception:
            try:
                df = pd.read_excel(file.name if hasattr(file, "name") else file)
            except Exception:
                return None, "Could not read file."
    elif demo_choice:
        key = DEMO_DATASETS.get(demo_choice)
        if not key:
            return None, "Select a dataset."
        df = _load_demo(key)
    else:
        return None, "Upload a CSV or select a demo dataset."

    cat_auto = [c for c in df.columns if df[c].dtype == "object" or df[c].nunique() < 15]
    info = (
        f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns\n\n"
        f"**Columns:** {', '.join(df.columns.tolist())}\n\n"
        f"**Auto-detected categorical:** {', '.join(cat_auto) if cat_auto else 'none'}\n\n"
        f"**Numeric columns:** {', '.join(df.select_dtypes(include=[np.number]).columns.tolist())}"
    )
    return df.head(20), info


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
DESCRIPTION = """
# TabGAN — Synthetic Tabular Data Generator

Generate high-quality synthetic tabular data using **GANs**, **Diffusion Models**, or simple **random sampling**.
Upload your own CSV or try a built-in dataset. Get instant **quality** and **privacy** reports.

[![GitHub](https://img.shields.io/github/stars/Diyago/Tabular-data-generation?style=social)](https://github.com/Diyago/Tabular-data-generation)
[![PyPI](https://img.shields.io/pypi/v/tabgan)](https://pypi.org/project/tabgan/)

```bash
pip install tabgan
```
"""

with gr.Blocks(
    title="TabGAN — Synthetic Tabular Data",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple"),
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Load Data")
            file_input = gr.File(label="Upload CSV / Excel", file_types=[".csv", ".xlsx", ".xls"])
            demo_dropdown = gr.Dropdown(
                choices=list(DEMO_DATASETS.keys()),
                label="Or select a demo dataset",
                value="Iris (150 rows, 4 features)",
            )
            preview_btn = gr.Button("Preview Data", variant="secondary")
            data_info = gr.Markdown()
            data_preview = gr.Dataframe(label="Data Preview", interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("### 2. Configure Generation")
            generator_dropdown = gr.Dropdown(
                choices=list(GENERATORS.keys()),
                value="GAN (CTGAN)",
                label="Generator Method",
            )
            gen_x_times = gr.Slider(
                minimum=0.5, maximum=5.0, value=1.1, step=0.1,
                label="Generation multiplier (gen_x_times)",
                info="How many synthetic rows to generate relative to original (1.0 = same count)",
            )
            epochs_slider = gr.Slider(
                minimum=5, maximum=300, value=30, step=5,
                label="Epochs (GAN only)",
                info="More epochs = better quality but slower",
            )
            cat_cols_input = gr.Textbox(
                label="Categorical columns (comma-separated, leave blank for auto-detect)",
                placeholder="e.g., gender, city, category",
            )
            target_col_input = gr.Textbox(
                label="Target column (optional, for ML utility evaluation)",
                placeholder="e.g., target, label, class",
            )
            generate_btn = gr.Button("Generate Synthetic Data", variant="primary", size="lg")

    status_box = gr.Markdown()

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 3. Results")
            output_table = gr.Dataframe(label="Synthetic Data (first 50 rows)", interactive=False)
            download_btn = gr.File(label="Download Full Synthetic CSV")

    with gr.Row():
        with gr.Column():
            output_plot = gr.Plot(label="Distribution Comparison")
        with gr.Column():
            metrics_output = gr.Markdown(label="Quality & Privacy Metrics")

    # --- Examples ---
    gr.Markdown("### Quick Examples")
    gr.Examples(
        examples=[
            [None, "Iris (150 rows, 4 features)", "GAN (CTGAN)", 1.1, 30, "", "target"],
            [None, "Wine (178 rows, 13 features)", "Forest Diffusion", 1.0, 30, "", "target"],
            [None, "California Housing (1000 rows, 8 features)", "GAN (CTGAN)", 1.0, 50, "", "MedHouseVal"],
        ],
        inputs=[file_input, demo_dropdown, generator_dropdown, gen_x_times, epochs_slider, cat_cols_input, target_col_input],
        label="Click an example to auto-fill settings",
    )

    gr.Markdown(
        "---\n"
        "**TabGAN** · [GitHub](https://github.com/Diyago/Tabular-data-generation) · "
        "[PyPI](https://pypi.org/project/tabgan/) · Apache 2.0 License"
    )

    # --- Event handlers ---
    preview_btn.click(
        fn=preview_data,
        inputs=[file_input, demo_dropdown],
        outputs=[data_preview, data_info],
    )

    # Auto-preview on dataset change
    demo_dropdown.change(
        fn=preview_data,
        inputs=[file_input, demo_dropdown],
        outputs=[data_preview, data_info],
    )

    file_input.change(
        fn=preview_data,
        inputs=[file_input, demo_dropdown],
        outputs=[data_preview, data_info],
    )

    generate_btn.click(
        fn=generate_synthetic,
        inputs=[file_input, demo_dropdown, generator_dropdown, gen_x_times, epochs_slider, cat_cols_input, target_col_input],
        outputs=[output_table, output_plot, metrics_output, download_btn, status_box],
    )


demo.launch(ssr_mode=False)
