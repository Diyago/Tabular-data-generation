[![CodeFactor](https://www.codefactor.io/repository/github/diyago/gan-for-tabular-data/badge)](https://www.codefactor.io/repository/github/diyago/gan-for-tabular-data)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://pepy.tech/badge/tabgan)](https://pepy.tech/project/tabgan)

# GANs for tabular  data

<img src="./images/tabular_gan.png" height="15%" width="15%">
We well know GANs for success in the realistic image generation. However, they can be applied in tabular data generation. We will review and examine some recent papers about tabular GANs in action.

* Arxiv article: ["Tabular GANs for uneven distribution"](https://arxiv.org/abs/2010.00638)
* Medium post: [GANs for tabular data](https://towardsdatascience.com/review-of-gans-for-tabular-data-a30a2199342)

### How to use library

* Installation: `pip install tabgan`
* To generate new data to train by sampling and then filtering by adversarial training
  call `GANGenerator().generate_data_pipe`:

``` python
from tabgan.sampler import OriginalGenerator, GANGenerator
import pandas as pd
import numpy as np

# random input data
train = pd.DataFrame(np.random.randint(-10, 150, size=(50, 4)), columns=list("ABCD"))
target = pd.DataFrame(np.random.randint(0, 2, size=(50, 1)), columns=list("Y"))
test = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))

# generate data
new_train1, new_target1 = OriginalGenerator().generate_data_pipe(train, target, test, )
new_train2, new_target2 = GANGenerator().generate_data_pipe(train, target, test, )

# example with all params defined
new_train3, new_target3 = GANGenerator(gen_x_times=1.1, cat_cols=None,
           bot_filter_quantile=0.001, top_filter_quantile=0.999, is_post_process=True,
           adversaial_model_params={
               "metrics": "AUC", "max_depth": 2, "max_bin": 100, 
               "learning_rate": 0.02, "random_state": 42, "n_estimators": 500,
           }, pregeneration_frac=2, only_generated_data=False,
           gan_params = {"batch_size": 500, "patience": 25, "epochs" : 500,}).generate_data_pipe(train, target,
                                          test, deep_copy=True, only_adversarial=False, use_adversarial=True)
```

Both samplers `OriginalGenerator` and `GANGenerator` have same input parameters:

* **gen_x_times**: float = 1.1 - how much data to generate, output might be less because of postprocessing and
  adversarial filtering
* **cat_cols**: list = None - categorical columns
* **bot_filter_quantile**: float = 0.001 - bottom quantile for postprocess filtering
* **top_filter_quantile**: float = 0.999 - bottom quantile for postprocess filtering
* **is_post_process**: bool = True - perform or not post-filtering, if false bot_filter_quantile and top_filter_quantile
  ignored
* **adversaial_model_params**: dict params for adversarial filtering model, default values for binary task
* **pregeneration_frac**: float = 2 - for generataion step gen_x_times * pregeneration_frac amount of data will
  generated. However in postprocessing (1 + gen_x_times) % of original data will be returned
* **gan_params**: dict params for GAN training

For `generate_data_pipe` methods params:

* **train_df**: pd.DataFrame Train dataframe which has separate target
* **target**: pd.DataFrame Input target for the train dataset
* **test_df**: pd.DataFrame Test dataframe - newly generated train dataframe should be close to it
* **deep_copy**: bool = True - make copy of input files or not. If not input dataframes will be overridden
* **only_adversarial**: bool = False - only adversarial fitering to train dataframe will be performed
* **use_adversarial**: bool = True - perform or not adversarial filtering
* **only_generated_data**: bool = False  - After generation get only newly generated, without 
  concating input train dataframe.  
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

### Datasets and experiment design

**Running experiment**

To run experiment follow these steps:

1. Clone the repository. All required dataset are stored in `./Research/data` folder
2. Install requirements `pip install -r requirements.txt`
4. Run all experiments  `python ./Research/run_experiment.py`. Run all experiments  `python run_experiment.py`. You may
   add more datasets, adjust validation type and categorical encoders.
5. Observe metrics across all experiment in console or in `./Research/results/fit_predict_scores.txt`

**Task formalization**

Let say we have **T_train** and **T_test** (train and test set respectively). We need to train the model on **T_train**
and make predictions on **T_test**. However, we will increase the train by generating new data by GAN, somehow similar
to **T_test**, without using ground truth labels.

**Experiment design**

Let say we have **T_train** and **T_test** (train and test set respectively). The size of **T_train** is smaller and
might have different data distribution. First of all, we train CTGAN on **T_train** with ground truth labels (step 1),
then generate additional data **T_synth** (step 2). Secondly, we train boosting in an adversarial way on concatenated **
T_train** and **T_synth** (target set to 0) with **T_test** (target set to 1) (steps 3 & 4). The goal is to apply newly
trained adversarial boosting to obtain rows more like **T_test**. Note - initial ground truth labels aren"t used for
adversarial training. As a result, we take top rows from **T_train** and **T_synth** sorted by correspondence to **
T_test** (steps 5 & 6), and train new boosting on them and check results on **T_test**.

![Experiment design and workflow](./images/workflow.png?raw=true)

**Picture 1.1** Experiment design and workflow

Of course for the benchmark purposes we will test ordinal training without these tricks and another original pipeline
but without CTGAN (in step 3 we won"t use **T_sync**).

**Datasets**

All datasets came from different domains. They have a different number of observations, number of categorical and
numerical features. The objective for all datasets - binary classification. Preprocessing of datasets were simple:
removed all time-based columns from datasets. Remaining columns were either categorical or numerical.

**Table 1.1** Used datasets

| Name | Total points | Train points | Test points | Number of features | Number of categorical features | Short description |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [Telecom](https://www.kaggle.com/blastchar/telco-customer-churn)   | 7.0k | 4.2k |  2.8k |  20   |  16  | Churn prediction for telecom data |
| [Adult](https://www.kaggle.com/wenruliu/adult-income-dataset)   | 48.8k | 29.3k | 19.5k  |  15  | 8 | Predict if persons" income is bigger 50k |
| [Employee](https://www.kaggle.com/c/amazon-employee-access-challenge/data)   | 32.7k | 19.6k | 13.1k  | 10  | 9 | Predict an employee"s access needs, given his/her job role|
| [Credit](https://www.kaggle.com/c/home-credit-default-risk/data)   | 307.5k | 184.5k | 123k  |  121  | 18 | Loan repayment |
| [Mortgages](https://www.crowdanalytix.com/contests/propensity-to-fund-mortgages)   |  45.6k | 27.4k | 18.2k | 20 | 9 | Predict if house mortgage is founded |
| [Taxi](https://www.crowdanalytix.com/contests/mckinsey-big-data-hackathon) | 892.5k | 535.5k | 357k | 8 | 5 | Predict the probability of an offer being accepted by a certain driver |
| [Poverty_A](https://www.drivendata.org/competitions/50/worldbank-poverty-prediction/page/99/)   | 37.6k | 22.5k | 15.0k | 41 | 38 | Predict whether or not a given household for a given country is poor or not |

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
| telecom                |           **0.995** |          0.868 |                      0.992 |

**Table 1.3** Different sampling results, higher is better for a mean (ROC AUC), lower is better for std (100% - maximum
per dataset ROC AUC)

| sample_type     |     mean |       std |
|:----------------|---------:|----------:|
| None            | 0.980 | 0.036 |
| gan             | 0.969 | 0.06 |
| sample_original | **0.981** | **0.032** |

**Table 1.4** same_target_prop is equal 1 then the target rate for train and test are different no more than 5%. Higher
is better.

| sample_type     |   same_target_prop |   prop_test_score |
|:----------------|-------------------:|------------------:|
| None            |                  0 |          0.964 |
| None            |                  1 |          0.985 |
| gan             |                  0 |          0.966 |
| gan             |                  1 |          0.945 |
| sample_original |                  0 |          0.973 |
| sample_original |                  1 |          0.984 |

## Acknowledgments

The author would like to thank Open Data Science community [7] for many valuable discussions and educational help in the
growing field of machine and deep learning. Also, special big thanks to Sber [8] for allowing solving such tasks and
providing computational resources.

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
library itself:
```bibtex
@misc{Diyago2020tabgan,
    author       = {Ashrapov, Insaf},
    title        = {GANs for tabular data},
    howpublished = {\url{https://github.com/Diyago/GAN-for-tabular-data}},
    year         = {2020}
}
```

## References

[1] Jonathan Hui. GAN — What is Generative Adversarial Networks GAN? (2018), medium article

[2]Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville,
Yoshua Bengio. Generative Adversarial Networks (2014). arXiv:1406.2661

[3] Lei Xu LIDS, Kalyan Veeramachaneni. Synthesizing Tabular Data using Generative Adversarial Networks (2018). arXiv:
1811.11264v1 [cs.LG]

[4] Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni. Modeling Tabular Data using Conditional
GAN (2019). arXiv:1907.00503v2 [cs.LG]

[5] Denis Vorotyntsev. Benchmarking Categorical Encoders (2019). Medium post

[6] Insaf Ashrapov. GAN-for-tabular-data (2020). Github repository.

[7] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila. Analyzing and Improving the
Image Quality of StyleGAN (2019) arXiv:1912.04958v2 [cs.CV]

[8]  ODS.ai: Open data science (2020), https://ods.ai/

[9]  Sber (2020), https://www.sberbank.ru/
