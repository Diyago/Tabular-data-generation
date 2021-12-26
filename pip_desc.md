[![CodeFactor](https://www.codefactor.io/repository/github/diyago/gan-for-tabular-data/badge)](https://www.codefactor.io/repository/github/diyago/gan-for-tabular-data)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
# GANs for tabular  data
<img src="https://raw.githubusercontent.com/Diyago/GAN-for-tabular-data/e5a4d437655261755de962b9779c73203611d921/images/logo%20tabular%20gan.svg" height="15%" width="15%">

We well know GANs for success in the realistic image generation. However, they can be applied in tabular data generation. We will review and examine some recent papers about tabular GANs in action.

* Github project: ["GAN-for-tabular-data"](https://github.com/Diyago/GAN-for-tabular-data)
* Arxiv article: ["Tabular GANs for uneven distribution"](https://arxiv.org/abs/2010.00638)
* Medium post: [GANs for tabular data](https://towardsdatascience.com/review-of-gans-for-tabular-data-a30a2199342)

### Library goal

Let say we have **T_train** and **T_test** (train and test set respectively).
We need to train the model on **T_train** and make predictions on **T_test**.
However, we will increase the train by generating new data by GAN,
somehow similar to **T_test**, without using ground truth labels.

### How to use library

* Installation: `pip install tabgan`
* To generate new data to train by sampling and then filtering by adversarial
  training call `GANGenerator().generate_data_pipe`:

``` python
from tabgan.sampler import OriginalGenerator, GANGenerator
import pandas as pd
import numpy as np

# random input data
train = pd.DataFrame(np.random.randint(-10, 150, size=(150, 4)), columns=list("ABCD"))
target = pd.DataFrame(np.random.randint(0, 2, size=(150, 1)), columns=list("Y"))
test = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))

# generate data
new_train1, new_target1 = OriginalGenerator().generate_data_pipe(train, target, test, )
new_train1, new_target1 = GANGenerator().generate_data_pipe(train, target, test, )

# example with all params defined
new_train3, new_target3 = GANGenerator(gen_x_times=1.1, cat_cols=None,
           bot_filter_quantile=0.001, top_filter_quantile=0.999, is_post_process=True,
           adversarial_model_params={
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
* **is_post_process**: bool = True - perform or not postfiltering, if false bot_filter_quantile
 and top_filter_quantile ignored
* **adversarial_model_params**: dict params for adversarial filtering model, default values for binary task
* **pregeneration_frac**: float = 2 - for generataion step gen_x_times * pregeneration_frac amount of data
will be generated. However, in postprocessing (1 + gen_x_times) % of original data will be returned
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

* **@return**: -> Tuple[pd.DataFrame, pd.DataFrame] -  Newly generated train dataframe and test data

Thus, you may use this library to improve your dataset quality:

``` python
def fit_predict(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    return sklearn.metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])


if __name__ == "__main__":
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

**Running experiment**

To run experiment follow these steps:
1. Clone the repository. All required dataset are stored in `./Research/data` folder
2. Install requirements `pip install -r requirements.txt`
4. Run all experiments  `python ./Research/run_experiment.py`. Run all experiments  `python run_experiment.py`. You may add more datasets, adjust validation type and categorical encoders.
5. Observe metrics across all experiment in console or
   in `./Research/results/fit_predict_scores.txt`



## Acknowledgments

The author would like to thank Open Data Science community [7] for many
valuable discussions and educational help in the growing field of machine and
deep learning. Also, special big thanks to Sber [8] for allowing solving
such tasks and providing computational resources.

## References

[1] Jonathan Hui. GAN — What is Generative Adversarial Networks GAN? (2018), medium article

[2]Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. Generative Adversarial Networks (2014). arXiv:1406.2661

[3] Lei Xu LIDS, Kalyan Veeramachaneni. Synthesizing Tabular Data using Generative Adversarial Networks (2018). arXiv:1811.11264v1 [cs.LG]

[4] Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni. Modeling Tabular Data using Conditional GAN (2019). arXiv:1907.00503v2 [cs.LG]

[5] Denis Vorotyntsev. Benchmarking Categorical Encoders (2019). Medium post

[6] Insaf Ashrapov. GAN-for-tabular-data (2020). Github repository.

[7] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila. Analyzing and Improving the Image Quality of StyleGAN (2019) arXiv:1912.04958v2 [cs.CV]

[8]  ODS.ai: Open data science (2020), https://ods.ai/

[9]  Sber (2020), https://www.sberbank.ru/
