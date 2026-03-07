import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


def test_tabgan_generate_cli_creates_output_with_target():
    # Prepare temporary input and output CSV paths
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "train.csv")
        output_path = os.path.join(tmpdir, "synthetic.csv")

        # Small dummy dataset with a target column
        df = pd.DataFrame(
            np.random.randint(0, 10, size=(20, 3)),
            columns=["A", "B", "target"],
        )
        df.to_csv(input_path, index=False)

        # Invoke the installed console script through Python -m to avoid PATH issues in CI
        cmd = [
            sys.executable,
            "-m",
            "tabgan.cli",
            "--input-csv",
            input_path,
            "--target-col",
            "target",
            "--generator",
            "original",
            "--gen-x-times",
            "1.0",
            "--output-csv",
            output_path,
        ]

        result = subprocess.run(cmd, check=False, capture_output=True, text=True)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(output_path), "Output CSV was not created by CLI"

        out_df = pd.read_csv(output_path)

        # Target column must be present
        assert "target" in out_df.columns
        # At least as many rows as the original (OriginalGenerator samples with replacement)
        assert len(out_df) >= len(df)

