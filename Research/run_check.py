import gc

from tabgan.sampler import OriginalGenerator, GANGenerator, ForestDiffusionGenerator, LLMGenerator
from sklearn.preprocessing import OrdinalEncoder

import warnings
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pandas as pd
from tabgan.utils import compare_dataframes
import logging

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*No further splits with positive gain.*")

params = {
    'objective': 'binary',  # Assuming binary classification
    'metric': 'auc',  # Track AUC during training
    'learning_rate': 0.1,
    'num_leaves': 31,  # Adjust based on data complexity
    'feature_fraction': 0.9,  # Randomly select features at each tree split
    'bagging_fraction': 0.8,  # Subsample of data for each tree
    'bagging_freq': 5,  # Subsample frequency
    'verbose': -1,  # Suppress training messages
    'early_stopping_rounds': 15,
    'verbosity': -1,
}


def g(df):
    for col in df.filter(like='cat'):
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.as_ordered()
    return df


train_prop_size = 0.35
dataset_list = [
    "telecom",
    "adult",
    "employee",
    "mortgages",
    "poverty_A",
    "credit",
    "taxi",
]
typegens = [
    "as is",
    "original",
    "gan",
    "diff",
]
# Create an empty list to store results dictionaries
results = []


def gen_data(train, target, test, typegen, cat_cols):
    if typegen == 'original':
        return OriginalGenerator(cat_cols=cat_cols).generate_data_pipe(train, target, test, )
    elif typegen == 'gan':
        return GANGenerator(cat_cols=cat_cols,
                            gen_params={"batch_size": 500, "epochs": 10, "patience": 5}).generate_data_pipe(
            train, target, test, )
    elif typegen == 'diff':
        return ForestDiffusionGenerator(cat_cols=cat_cols).generate_data_pipe(train, target, test, )
    elif typegen == 'llm':
        return LLMGenerator(cat_cols=cat_cols, gen_params={"batch_size": 32, "epochs": 4, "llm": "distilgpt2",
                                                           "max_length": 500}).generate_data_pipe(train, target, test, )
    elif typegen == 'as is':
        return train, target
    else:
        raise ValueError("Not found given typegen")


def fillna(df):
    return df.fillna(df.apply(lambda x: x.mean() if x.dtype == 'int64' or x.dtype == 'float64' else x.mode().iloc[0]))


if __name__ == "__main__":
    for dataset_name in tqdm(dataset_list):
        dataset_pth = rf".\Research\data\{dataset_name}\{dataset_name}.gz"

        # Load processed dataset
        data = pd.read_csv(dataset_pth)
        data = g(data)
        data = fillna(data)
        cat_cols = [col for col in data.columns if col.startswith("cat")]

        X_train, X_test, y_train, y_test = train_test_split(
            data.drop("target", axis=1),
            data["target"],
            test_size=0.6,
            shuffle=False,
            random_state=42,
        )

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=1)
        X_test, y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)

        train_size = X_train.shape[0]
        X_train = X_train.head(int(train_size * train_prop_size)).reset_index(drop=True)
        y_train = y_train.head(int(train_size * train_prop_size)).reset_index(drop=True)

        for typegen in tqdm(typegens):
            if typegen == "diff":
                le = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
                X_train[cat_cols] = le.fit_transform(X_train[cat_cols])
                X_test[cat_cols] = le.transform(X_test[cat_cols])
                X_val[cat_cols] = le.transform(X_val[cat_cols])
            X_train, y_train = gen_data(pd.DataFrame(X_train.reset_index(drop=True)).copy(),
                                        pd.DataFrame(y_train.astype(int).reset_index(drop=True)).copy(),
                                        pd.DataFrame(X_test.reset_index(drop=True)).copy(), typegen=typegen,
                                        cat_cols=cat_cols)

            lgb.set_logger(logging.ERROR)

            # Create LightGBM datasets (separate training and testing data)
            lgb_train = lgb.Dataset(X_train, label=y_train, params={"verbosity": -1})
            lgb_val = lgb.Dataset(X_val, label=y_val, params={"verbosity": -1})

            # Train the LightGBM model
            model = lgb.train(params, lgb_train, valid_sets=[lgb_val])
            y_pred = model.predict(X_test)
            roc_auc = roc_auc_score(y_test, y_pred)
            dataset_results = {"dataset": dataset_name, "roc_auc": roc_auc, "gen_type": typegen,
                               "gen_quality": compare_dataframes(X_test, X_train)}
            results.append(dataset_results)
            gc.collect()
    results_df = pd.DataFrame(results)
    print(results_df)
