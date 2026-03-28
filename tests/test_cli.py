import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


def _make_cli_env():
    """Return an env dict with PYTHONPATH pointing at the src directory."""
    env = os.environ.copy()
    src_path = os.path.join(os.path.dirname(__file__), "..", "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_cli(args, env):
    """Run `python -m tabgan.cli` with the given extra arguments."""
    cmd = [sys.executable, "-m", "tabgan.cli"] + args
    return subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)


def test_tabgan_generate_cli_creates_output_with_target():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "train.csv")
        output_path = os.path.join(tmpdir, "synthetic.csv")

        df = pd.DataFrame(
            np.random.randint(0, 10, size=(20, 3)),
            columns=["A", "B", "target"],
        )
        df.to_csv(input_path, index=False)

        env = _make_cli_env()
        result = _run_cli([
            "--input-csv", input_path,
            "--target-col", "target",
            "--generator", "original",
            "--gen-x-times", "1.0",
            "--output-csv", output_path,
        ], env)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(output_path), "Output CSV was not created by CLI"

        out_df = pd.read_csv(output_path)
        assert "target" in out_df.columns
        assert len(out_df) >= len(df)


def test_cli_gan_generator():
    """CLI with --generator gan produces valid output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "train.csv")
        output_path = os.path.join(tmpdir, "synthetic.csv")

        df = pd.DataFrame(
            np.random.randint(0, 50, size=(30, 3)),
            columns=["A", "B", "target"],
        )
        df.to_csv(input_path, index=False)

        env = _make_cli_env()
        result = _run_cli([
            "--input-csv", input_path,
            "--target-col", "target",
            "--generator", "gan",
            "--gen-x-times", "1.0",
            "--output-csv", output_path,
        ], env)

        assert result.returncode == 0, f"CLI (gan) failed: {result.stderr}"
        assert os.path.exists(output_path)

        out_df = pd.read_csv(output_path)
        assert "target" in out_df.columns
        assert len(out_df) > 0


def test_cli_only_generated_flag():
    """CLI with --only-generated returns output without original rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "train.csv")
        output_path = os.path.join(tmpdir, "synthetic.csv")

        df = pd.DataFrame(
            np.random.randint(0, 10, size=(20, 3)),
            columns=["A", "B", "target"],
        )
        df.to_csv(input_path, index=False)

        env = _make_cli_env()
        result = _run_cli([
            "--input-csv", input_path,
            "--target-col", "target",
            "--generator", "original",
            "--gen-x-times", "1.0",
            "--only-generated",
            "--output-csv", output_path,
        ], env)

        assert result.returncode == 0, f"CLI (only-generated) failed: {result.stderr}"
        assert os.path.exists(output_path)

        out_df = pd.read_csv(output_path)
        assert "target" in out_df.columns
        assert len(out_df) > 0


def test_cli_without_target():
    """CLI without --target-col should work (target=None path)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "train.csv")
        output_path = os.path.join(tmpdir, "synthetic.csv")

        df = pd.DataFrame(
            np.random.randint(0, 10, size=(20, 3)),
            columns=["A", "B", "C"],
        )
        df.to_csv(input_path, index=False)

        env = _make_cli_env()
        result = _run_cli([
            "--input-csv", input_path,
            "--generator", "original",
            "--gen-x-times", "1.0",
            "--output-csv", output_path,
        ], env)

        assert result.returncode == 0, f"CLI (no target) failed: {result.stderr}"
        assert os.path.exists(output_path)

        out_df = pd.read_csv(output_path)
        assert len(out_df) > 0


def test_cli_with_cat_cols():
    """CLI with --cat-cols passes categorical column names correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "train.csv")
        output_path = os.path.join(tmpdir, "synthetic.csv")

        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "num": rng.randint(0, 100, 30),
            "cat": rng.choice(["X", "Y", "Z"], 30),
            "target": rng.randint(0, 2, 30),
        })
        df.to_csv(input_path, index=False)

        env = _make_cli_env()
        result = _run_cli([
            "--input-csv", input_path,
            "--target-col", "target",
            "--generator", "original",
            "--gen-x-times", "1.0",
            "--cat-cols", "cat",
            "--output-csv", output_path,
        ], env)

        assert result.returncode == 0, f"CLI (cat-cols) failed: {result.stderr}"
        assert os.path.exists(output_path)

        out_df = pd.read_csv(output_path)
        assert "cat" in out_df.columns
        assert "target" in out_df.columns

