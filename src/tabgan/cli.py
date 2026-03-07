import argparse
import logging
from typing import List, Optional

import pandas as pd

from tabgan.sampler import (
    OriginalGenerator,
    GANGenerator,
    ForestDiffusionGenerator,
    LLMGenerator,
)


def _parse_cat_cols(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [c.strip() for c in raw.split(",") if c.strip()]


def main() -> None:
    """
    Command-line interface for generating synthetic tabular data with tabgan.

    Example:
        tabgan-generate \\
            --input-csv train.csv \\
            --target-col target \\
            --generator gan \\
            --gen-x-times 1.5 \\
            --cat-cols year,gender \\
            --output-csv synthetic_train.csv
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic tabular data using tabgan samplers."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to input CSV file containing training data (with or without target column).",
    )
    parser.add_argument(
        "--target-col",
        default=None,
        help="Name of the target column in the CSV (optional).",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to write the generated synthetic dataset as CSV.",
    )
    parser.add_argument(
        "--generator",
        choices=["original", "gan", "diffusion", "llm"],
        default="gan",
        help="Which sampler to use for generation.",
    )
    parser.add_argument(
        "--gen-x-times",
        type=float,
        default=1.1,
        help="Factor controlling how many synthetic samples to generate relative to the training size.",
    )
    parser.add_argument(
        "--cat-cols",
        default=None,
        help="Comma-separated list of categorical column names (e.g. 'year,gender').",
    )
    parser.add_argument(
        "--only-generated",
        action="store_true",
        help="If set, output only synthetic rows instead of original + synthetic.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info("Reading input CSV from %s", args.input_csv)
    df = pd.read_csv(args.input_csv)

    target_df = None
    train_df = df
    if args.target_col is not None:
        if args.target_col not in df.columns:
            raise ValueError(f"Target column '{args.target_col}' not found in input CSV.")
        target_df = df[[args.target_col]]
        train_df = df.drop(columns=[args.target_col])

    cat_cols = _parse_cat_cols(args.cat_cols)

    generator_map = {
        "original": OriginalGenerator,
        "gan": GANGenerator,
        "diffusion": ForestDiffusionGenerator,
        "llm": LLMGenerator,
    }
    generator_cls = generator_map[args.generator]

    logging.info("Initializing %s generator", generator_cls.__name__)
    generator = generator_cls(
        gen_x_times=args.gen_x_times,
        cat_cols=cat_cols,
        only_generated_data=bool(args.only_generated),
    )

    # Use train_df itself as test_df when a dedicated hold-out set is not provided.
    logging.info("Generating synthetic data...")
    new_train, new_target = generator.generate_data_pipe(
        train_df, target_df, train_df
    )

    if new_target is not None and args.target_col is not None:
        out_df = new_train.copy()
        # new_target can be DataFrame or Series; align to a 1D array
        if hasattr(new_target, "values") and new_target.ndim > 1:
            out_df[args.target_col] = new_target.values.ravel()
        else:
            out_df[args.target_col] = new_target
    else:
        out_df = new_train

    logging.info("Writing synthetic data to %s", args.output_csv)
    out_df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()

