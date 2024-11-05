[![CodeFactor](https://www.codefactor.io/repository/github/diyago/gan-for-tabular-data/badge)](https://www.codefactor.io/repository/github/diyago/gan-for-tabular-data)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://pepy.tech/badge/tabgan)](https://pepy.tech/project/tabgan)

# GANs and TimeGANs, Diffusions, LLM for tabular  data

<img src="./images/tabular_gan.png" height="15%" width="15%">

Generative  Networks are well-known for their success in realistic image generation. However, they can also be applied to generate tabular data. We introduce major improvements for generating high-fidelity tabular data giving oppotunity to try GANS, TimeGANs, Diffusions and LLM for tabular data generations. 
* Arxiv article: ["Tabular GANs for uneven distribution"](https://arxiv.org/abs/2010.00638)
* Medium post: [GANs for tabular data](https://towardsdatascience.com/review-of-gans-for-tabular-data-a30a2199342)

## How to use library

* Installation: `pip install tabgan`
* To generate new data to train by sampling and then filtering by adversarial training
  call `GANGenerator().generate_data_pipe`:

### Data Format

TabGAN accepts data as a ```numpy.ndarray``` or ```pandas.DataFrame``` with columns categorized as:

* **Continuous Columns**: Numerical columns with any possible value.
* **Discrete Columns**: Columns with a limited set of values (e.g., categorical data).

Note: TabGAN does not differentiate between floats and integers, so all values are treated as floats. For integer requirements, round the output outside of TabGAN.

### Example code 

``` python
from tabgan.sampler import OriginalGenerator, GANGenerator, ForestDiffusionGenerator, LLMGenerator
import pandas as pd
import numpy as np


# random input data
train = pd.DataFrame(np.random.randint(-10, 150, size=(150, 4)), columns=list("ABCD"))
target = pd.DataFrame(np.random.randint(0, 2, size=(150, 1)), columns=list("Y"))
test = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))

# generate data
new_train1, new_target1 = OriginalGenerator().generate_data_pipe(train, target, test, )
new_train2, new_target2 = GANGenerator(gen_params={"batch_size": 500, "epochs": 10, "patience": 5 }).generate_data_pipe(train, target, test, )
new_train3, new_target3 = ForestDiffusionGenerator().generate_data_pipe(train, target, test, )
new_train4, new_target4 = LLMGenerator(gen_params={"batch_size": 32, 
                                                          "epochs": 4, "llm": "distilgpt2", "max_length": 500}).generate_data_pipe(train, target, test, )

# example with all params defined
new_train4, new_target4 = GANGenerator(gen_x_times=1.1, cat_cols=None,
           bot_filter_quantile=0.001, top_filter_quantile=0.999, is_post_process=True,
           adversarial_model_params={
               "metrics": "AUC", "max_depth": 2, "max_bin": 100, 
               "learning_rate": 0.02, "random_state": 42, "n_estimators": 100,
           }, pregeneration_frac=2, only_generated_data=False,
           gen_params = {"batch_size": 500, "patience": 25, "epochs" : 500,}).generate_data_pipe(train, target,
                                          test, deep_copy=True, only_adversarial=False, use_adversarial=True)
```

All samplers `OriginalGenerator`, `ForestDiffusionGenerator`, `LLMGenerator` and `GANGenerator` have same input parameters.

1. **GANGenerator** based on **CTGAN**
2. **ForestDiffusionGenerator** based on **Forest Diffusion (Tabular Diffusion and Flow-Matching)**
2. **LLMGenerator** based on **Language Models are Realistic Tabular Data Generators (GReaT framework)**

* **gen_x_times**: float = 1.1 - how much data to generate, output might be less because of postprocessing and
  adversarial filtering
* **cat_cols**: list = None - categorical columns
* **bot_filter_quantile**: float = 0.001 - bottom quantile for postprocess filtering
* **top_filter_quantile**: float = 0.999 - top quantile for postprocess filtering
* **is_post_process**: bool = True - perform or not post-filtering, if false bot_filter_quantile and top_filter_quantile
  ignored
* **adversarial_model_params**: dict params for adversarial filtering model, default values for binary task
* **pregeneration_frac**: float = 2 - for generation step gen_x_times * pregeneration_frac amount of data will
  be generated. However, in postprocessing (1 + gen_x_times) % of original data will be returned
* **gen_params**: dict params for GAN training

For `generate_data_pipe` methods params:

* **train_df**: pd.DataFrame Train dataframe which has separate target
* **target**: pd.DataFrame Input target for the train dataset
* **test_df**: pd.DataFrame Test dataframe - newly generated train dataframe should be close to it
* **deep_copy**: bool = True - make copy of input files or not. If not input dataframes will be overridden
* **only_adversarial**: bool = False - only adversarial filtering to train dataframe will be performed
* **use_adversarial**: bool = True - perform or not adversarial filtering
* **only_generated_data**: bool = False  - After generation get only newly generated, without 
  concatenating input train dataframe.  
* **@return**: -> Tuple[pd.DataFrame, pd.DataFrame] - Newly generated train dataframe and test data

Thus, you may use this library to improve your dataset quality:

``` python
def fit_predict(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    return sklearn.metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])



dataset = sklearn.datasets.load_breast_cancer()
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=25, max_depth=6)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    pd.DataFrame(dataset.data), pd.DataFrame(dataset.target, columns=["target"]), test_size=0.33, random_state=42)
print("initial metric", fit_predict(clf, X_train, y_train, X_test, y_test))

new_train1, new_target1 = OriginalGenerator().generate_data_pipe(X_train, y_train, X_test, )
print("OriginalGenerator metric", fit_predict(clf, new_train1, new_target1, X_test, y_test))

new_train1, new_target1 = GANGenerator().generate_data_pipe(X_train, y_train, X_test, )
print("GANGenerator metric", fit_predict(clf, new_train1, new_target1, X_test, y_test))
```
## Timeseries GAN generation TimeGAN

You can easily adjust code to generate multidimensional timeseries data.
Basically it extracts days, months and year from _date_. Demo how to use in the example below:
```python
import pandas as pd
import numpy as np
from tabgan.utils import get_year_mnth_dt_from_date,make_two_digit,collect_dates
from tabgan.sampler import OriginalGenerator, GANGenerator


train_size = 100
train = pd.DataFrame(
        np.random.randint(-10, 150, size=(train_size, 4)), columns=list("ABCD")
    )
min_date = pd.to_datetime('2019-01-01')
max_date = pd.to_datetime('2021-12-31')
d = (max_date - min_date).days + 1

train['Date'] = min_date + pd.to_timedelta(pd.np.random.randint(d, size=train_size), unit='d')
train = get_year_mnth_dt_from_date(train, 'Date')

new_train, new_target = GANGenerator(gen_x_times=1.1, cat_cols=['year'], bot_filter_quantile=0.001,
                                     top_filter_quantile=0.999,
                                     is_post_process=True, pregeneration_frac=2, only_generated_data=False).\
                                     generate_data_pipe(train.drop('Date', axis=1), None,
                                                        train.drop('Date', axis=1)
                                                                    )
new_train = collect_dates(new_train)
```

## Experiments
### Datasets and experiment design

**Check for data generation quality**
Just use built-in function
```
compare_dataframes(original_df, generated_df) # return between 0 and 1
```
**Running experiment**

To run experiment follow these steps:

1. Clone the repository. All required dataset are stored in `./Research/data` folder
2. Install requirements `pip install -r requirements.txt`
4. Run all experiments  `python ./Research/run_experiment.py`. Run all experiments  `python run_experiment.py`. You may
   add more datasets, adjust validation type and categorical encoders.
5. Observe metrics across all experiment in console or in `./Research/results/fit_predict_scores.txt`


**Experiment design**

![Experiment design and workflow](./images/workflow.png?raw=true)

**Picture 1.1** Experiment design and workflow

## Results
To determine the best sampling strategy, ROC AUC scores of each dataset were scaled (min-max scale) and then averaged
among the dataset.

**Table 1.2** Different sampling results across the dataset, higher is better (100% - maximum per dataset ROC AUC)

| dataset_name  |   None |   gan |   sample_original |
|:-----------------------|-------------------:|------------------:|------------------------------:|
| credit                 |           0.997 |          **0.998** |                      0.997 |
| employee               |           **0.986** |          0.966 |                      0.972 |
| mortgages              |           0.984 |          0.964 |                      **0.988** |
| poverty_A              |           0.937 |          **0.950** |                      0.933 |
| taxi                   |           0.966 |          0.938 |                      **0.987** |
| adult                  |           0.995 |          0.967 |                      **0.998** |

## Citation

If you use **GAN-for-tabular-data** in a scientific publication, we would appreciate references to the following BibTex entry:
arxiv publication:
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

[1] Lei Xu LIDS, Kalyan Veeramachaneni. Synthesizing Tabular Data using Generative Adversarial Networks (2018). arXiv:
1811.11264v1 [cs.LG]

[2] Alexia Jolicoeur-Martineau and Kilian Fatras and Tal Kachman. Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees ((2023) https://github.com/SamsungSAILMontreal/ForestDiffusion [cs.LG]

[3] Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni. Modeling Tabular data using Conditional GAN. NeurIPS, (2019)

[4] Vadim Borisov and Kathrin Sessler and Tobias Leemann and Martin Pawelczyk and Gjergji Kasneci. Language Models are Realistic Tabular Data Generators. ICLR, (2023)
