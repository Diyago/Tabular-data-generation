[![CodeFactor](https://www.codefactor.io/repository/github/diyago/tabular-data-generation/badge)](https://www.codefactor.io/repository/github/diyago/tabular-data-generation)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/pypi/pyversions/tabgan)](https://pypi.org/project/tabgan/)
[![PyPI Version](https://img.shields.io/pypi/v/tabgan.svg)](https://pypi.org/project/tabgan/)
[![Downloads](https://pepy.tech/badge/tabgan)](https://pepy.tech/project/tabgan)
[![CodeQL](https://github.com/diyago/Tabular-data-generation/workflows/CodeQL/badge.svg)](https://github.com/diyago/Tabular-data-generation/actions/workflows/codeql.yml)

# TabGAN - Synthetic Tabular Data Generation

A powerful library for generating high-quality synthetic tabular data using state-of-the-art generative models including GANs, Diffusion models, and Large Language Models.

<img src="images/tabular_gan.png" height="15%" width="15%">

## Overview

TabGAN is a comprehensive Python library that provides a unified interface for generating high-quality synthetic tabular data. It integrates multiple state-of-the-art generative approaches to address diverse data synthesis requirements:

- **GANs**: Conditional Tabular GAN (CTGAN) for modeling complex multivariate distributions with mixed data types
- **Diffusion Models**: Forest Diffusion for high-fidelity synthetic data generation with tree-based gradient boosting
- **LLMs**: GReaT (Generative Realistic Tabular data) framework leveraging language models for realistic tabular data synthesis
- **Time-Series**: TimeGAN support for temporal data generation preserving sequential dependencies

*Related Research: [Tabular GANs for uneven distribution (arXiv:2010.00638)](https://arxiv.org/abs/2010.00638)*

## Key Features

- **Multiple Generative Architectures**: Seamlessly switch between GANs, Diffusion Models, and LLMs via a unified API
- **Adversarial Filtering**: Built-in adversarial validation to ensure synthetic data preserves predictive utility
- **Mixed Data Type Support**: Native handling of continuous, categorical, and text columns
- **Conditional Generation**: Generate data conditioned on specific column values or distributions
- **Scalable Processing**: Efficient batch processing for large-scale datasets
- **Quality Validation**: Integrated metrics for comparing synthetic against original data distributions

## Installation

```bash
pip install tabgan
```

## Quick Start

```python
import pandas as pd
import numpy as np
from tabgan.sampler import GANGenerator

# Create sample data
train = pd.DataFrame(np.random.randint(-10, 150, size=(150, 4)), columns=list("ABCD"))
target = pd.DataFrame(np.random.randint(0, 2, size=(150, 1)), columns=list("Y"))
test = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))

# Generate synthetic data
new_train, new_target = GANGenerator().generate_data_pipe(train, target, test)
```

## Available Generators

| Generator | Description | Best For |
|-----------|-------------|----------|
| `GANGenerator` | CTGAN-based generation | General tabular data with mixed types |
| `ForestDiffusionGenerator` | Diffusion models + tree-based methods | Complex tabular structures |
| `LLMGenerator` | Large Language Model based | Capturing complex dependencies |
| `OriginalGenerator` | Baseline sampler | Baseline comparisons |

## API Reference

### Sampler Parameters

All generators accept these common parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gen_x_times` | float | 1.1 | Multiplier for data generation amount |
| `cat_cols` | list | None | Column names to treat as categorical |
| `bot_filter_quantile` | float | 0.001 | Bottom quantile for post-process filtering |
| `top_filter_quantile` | float | 0.999 | Top quantile for post-process filtering |
| `is_post_process` | bool | True | Enable post-filtering |
| `pregeneration_frac` | float | 2 | Pre-generation data multiplier |
| `only_generated_data` | bool | False | Return only generated data (no original) |
| `gen_params` | dict | See below | Generator-specific parameters |

### Generator-Specific Parameters

**GANGenerator:**
```python
{"batch_size": 500, "patience": 25, "epochs": 500}
```

**LLMGenerator:**
```python
{"batch_size": 32, "epochs": 4, "llm": "distilgpt2", "max_length": 500}
```

### generate_data_pipe Method

| Parameter | Type | Description |
|-----------|------|-------------|
| `train_df` | pd.DataFrame | Training features |
| `target` | pd.DataFrame | Target variable |
| `test_df` | pd.DataFrame | Test features |
| `deep_copy` | bool | Make copy of input dataframes |
| `only_adversarial` | bool | Only perform adversarial filtering |
| `use_adversarial` | bool | Enable adversarial filtering |

**Returns:** `Tuple[pd.DataFrame, pd.DataFrame]` - (new_train, new_target)

## Data Format

TabGAN accepts both `numpy.ndarray` and `pandas.DataFrame` inputs, supporting:

- **Continuous Variables**: Numerical columns with any real-valued data
- **Categorical Variables**: Discrete columns with a finite set of possible values

> **Note:** TabGAN internally processes all values as floating-point numbers. For integer-valued outputs, apply rounding after generation.

## Examples

### Basic Usage

```python
from tabgan.sampler import OriginalGenerator, GANGenerator, ForestDiffusionGenerator, LLMGenerator
import pandas as pd
import numpy as np

train = pd.DataFrame(np.random.randint(-10, 150, size=(150, 4)), columns=list("ABCD"))
target = pd.DataFrame(np.random.randint(0, 2, size=(150, 1)), columns=list("Y"))
test = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))

# Different generators
new_train1, new_target1 = OriginalGenerator().generate_data_pipe(train, target, test)
new_train2, new_target2 = GANGenerator(gen_params={"batch_size": 500, "epochs": 10, "patience": 5}).generate_data_pipe(train, target, test)
new_train3, new_target3 = ForestDiffusionGenerator().generate_data_pipe(train, target, test)
new_train4, new_target4 = LLMGenerator(gen_params={"batch_size": 32, "epochs": 4, "llm": "distilgpt2", "max_length": 500}).generate_data_pipe(train, target, test)
```

### Full Parameter Example

```python
new_train, new_target = GANGenerator(
    gen_x_times=1.1,
    cat_cols=None,
    bot_filter_quantile=0.001,
    top_filter_quantile=0.999,
    is_post_process=True,
    adversarial_model_params={
        "metrics": "AUC", "max_depth": 2, "max_bin": 100,
        "learning_rate": 0.02, "random_state": 42, "n_estimators": 100,
    },
    pregeneration_frac=2,
    only_generated_data=False,
    gen_params={"batch_size": 500, "patience": 25, "epochs": 500}
).generate_data_pipe(
    train, target, test,
    deep_copy=True,
    only_adversarial=False,
    use_adversarial=True
)
```

### LLM Conditional Text Generation

Generate synthetic data with LLMs while controlling text generation based on specific conditions. This uses the internal `_generate_via_prompt` method for novel text generation.

```python
import pandas as pd
from tabgan.sampler import LLMGenerator

# Create sample data with text and categorical columns
train = pd.DataFrame({
    "Name": ["Anna", "Maria", "Ivan", "Sergey", "Olga", "Boris"],
    "Gender": ["F", "F", "M", "M", "F", "M"],
    "Age": [25, 30, 35, 40, 28, 32],
    "Occupation": ["Engineer", "Doctor", "Artist", "Teacher", "Manager", "Pilot"]
})

# Generate new names conditioned on Gender, with other features imputed
new_train, _ = LLMGenerator(
    gen_x_times=1.5,  # Generate 1.5x the original data
    text_generating_columns=["Name"],  # Generate novel names
    conditional_columns=["Gender"],    # Condition on Gender column
    gen_params={"batch_size": 32, "epochs": 4, "llm": "distilgpt2", "max_length": 500},
    is_post_process=False  # Disable post-processing for this example
).generate_data_pipe(
    train,
    target=None,
    test_df=None,
    only_generated_data=True  # Return only generated data
)

print(new_train)
```

**Parameters for conditional generation:**
- `text_generating_columns`: List of column names to generate novel text for
- `conditional_columns`: List of column names that condition the text generation

The model will:
1. Sample values for conditional columns from their distributions
2. Impute remaining non-text columns using the LLM
3. Generate novel text for text columns via prompt-based generation (using `_generate_via_prompt`)
4. Ensure generated text values are unique (not present in original data)

### LLM API-Based Text Generation

Use external LLM APIs (LM Studio, OpenAI, Ollama) for text generation instead of local models. This allows you to leverage powerful models running on remote servers or local API endpoints.

```python
import pandas as pd
from tabgan.sampler import LLMGenerator
from tabgan.llm_config import LLMAPIConfig

# Create sample data
train = pd.DataFrame({
    "Name": ["Anna", "Maria", "Ivan", "Sergey", "Olga", "Boris"],
    "Gender": ["F", "F", "M", "M", "F", "M"],
    "Age": [25, 30, 35, 40, 28, 32],
    "Occupation": ["Engineer", "Doctor", "Artist", "Teacher", "Manager", "Pilot"]
})

# Configure API connection (LM Studio example)
api_config = LLMAPIConfig.from_lm_studio(
    base_url="http://localhost:1234",
    model="google/gemma-3-12b",
    timeout=90
)

# Or use OpenAI
# api_config = LLMAPIConfig.from_openai(
#     api_key="your-api-key",
#     model="gpt-4"
# )

# Or use Ollama
# api_config = LLMAPIConfig.from_ollama(
#     base_url="http://localhost:11434",
#     model="llama3"
# )

# Generate with API-based text generation
new_train, _ = LLMGenerator(
    gen_x_times=1.5,
    text_generating_columns=["Name"],
    conditional_columns=["Gender"],
    gen_params={"batch_size": 32, "epochs": 4, "llm": "distilgpt2", "max_length": 500},
    llm_api_config=api_config,  # Use external API for text generation
    is_post_process=False
).generate_data_pipe(
    train,
    target=None,
    test_df=None,
    only_generated_data=True
)

print(new_train)
```

**Configuration Options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | str | `"http://localhost:1234"` | Base URL for the API server |
| `model` | str | `"google/gemma-3-12b"` | Model identifier to use |
| `api_key` | str | None | API key for authentication (OpenAI, etc.) |
| `timeout` | int | 90 | Request timeout in seconds |
| `max_tokens` | int | 256 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0.0-2.0) |
| `system_prompt` | str | None | System prompt to guide generation |

**Supported API Providers:**
- **LM Studio**: Local LLM server with OpenAI-compatible API
- **OpenAI**: GPT-4, GPT-3.5, and other OpenAI models
- **Ollama**: Local LLM server for running open-source models
- **Any OpenAI-compatible API**: Custom endpoints with compatible schema

**Testing API Connection:**

```python
from tabgan.llm_config import LLMAPIConfig
from tabgan.llm_api_client import LLMAPIClient

# Test if API is accessible
config = LLMAPIConfig.from_lm_studio()
with LLMAPIClient(config) as client:
    is_connected = client.check_connection()
    print(f"API available: {is_connected}")
    
    # Generate text directly
    text = client.generate("Generate a female name: ")
    print(f"Generated: {text}")
```

### Improving Model Performance

```python
import sklearn
from tabgan.sampler import GANGenerator

def evaluate(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    return sklearn.metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

# Load dataset
dataset = sklearn.datasets.load_breast_cancer()
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=25, max_depth=6)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    pd.DataFrame(dataset.data), pd.DataFrame(dataset.target, columns=["target"]),
    test_size=0.33, random_state=42)

# Compare performance
print("Baseline:", evaluate(clf, X_train, y_train, X_test, y_test))

# Generate and evaluate
new_train, new_target = GANGenerator().generate_data_pipe(X_train, y_train, X_test)
print("With GAN:", evaluate(clf, new_train, new_target, X_test, y_test))
```

### Time-Series Data Generation

```python
import pandas as pd
import numpy as np
from tabgan.utils import get_year_mnth_dt_from_date, collect_dates
from tabgan.sampler import GANGenerator

train = pd.DataFrame(np.random.randint(-10, 150, size=(100, 4)), columns=list("ABCD"))
min_date, max_date = pd.to_datetime('2019-01-01'), pd.to_datetime('2021-12-31')
d = (max_date - min_date).days + 1
train['Date'] = min_date + pd.to_timedelta(np.random.randint(d, size=100), unit='d')
train = get_year_mnth_dt_from_date(train, 'Date')

new_train, _ = GANGenerator(
    gen_x_times=1.1, cat_cols=['year'], bot_filter_quantile=0.001,
    top_filter_quantile=0.999, is_post_process=True, pregeneration_frac=2,
    only_generated_data=False
).generate_data_pipe(
    train.drop('Date', axis=1), None, train.drop('Date', axis=1)
)
new_train = collect_dates(new_train)
```

## Data Quality Validation

Validate the statistical fidelity of generated data using the built-in evaluation utilities:

```python
from tabgan.utils import compare_dataframes

# Returns a quality score between 0 (low fidelity) and 1 (high fidelity)
quality_score = compare_dataframes(original_df, generated_df)
```

### Experimental Workflow

![Experiment design and workflow](images/workflow.png)

## Benchmark Results

The following table shows normalized ROC AUC scores (higher is better):

| Dataset | None | GAN | Sample Original |
|---------|------|-----|-----------------|
| credit | 0.997 | **0.998** | 0.997 |
| employee | **0.986** | 0.966 | 0.972 |
| mortgages | 0.984 | 0.964 | **0.988** |
| poverty_A | 0.937 | **0.950** | 0.933 |
| taxi | 0.966 | 0.938 | **0.987** |
| adult | 0.995 | 0.967 | **0.998** |

## Citation

If you use TabGAN in your research, please cite:

```bibtex
@misc{ashrapov2020tabular,
      title={Tabular GANs for uneven distribution},
      author={Insaf Ashrapov},
      year={2020},
      eprint={2010.00638},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## References

[1] Xu, L., & Veeramachaneni, K. (2018). *Synthesizing Tabular Data using Generative Adversarial Networks*. arXiv:1811.11264 [cs.LG].

[2] Jolicoeur-Martineau, A., Fatras, K., & Kachman, T. (2023). *Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees*. SamsungSAILMontreal/ForestDiffusion.

[3] Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). *Modeling Tabular data using Conditional GAN*. NeurIPS.

[4] Borisov, V., Sessler, K., Leemann, T., Pawelczyk, M., & Kasneci, G. (2023). *Language Models are Realistic Tabular Data Generators*. ICLR.

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.
