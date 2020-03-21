import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import Model
from utils import save_exp_to_file, extend_gan_train, extend_from_original


def execute_experiment(dataset_name, encoders_list, validation_type, sample_type=None):
    dataset_pth = f"./data/{dataset_name}/{dataset_name}.gz"
    results = {}

    # load processed dataset
    data = pd.read_csv(dataset_pth)
    data.fillna(data.mean(), inplace=True)

    for train_prop_size in [0.05, 0.1, 0.25, 0.5, 0.75]:
        # make train-test split
        cat_cols = [col for col in data.columns if col.startswith("cat")]
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop("target", axis=1),
            data["target"],
            test_size=0.6,
            shuffle=False,
            random_state=42,
        )
        X_test, y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)

        train_size = X_train.shape[0]
        X_train = X_train.head(int(train_size * train_prop_size)).reset_index(drop=True)
        y_train = y_train.head(int(train_size * train_prop_size)).reset_index(drop=True)
        mean_target_before_sampling_train = np.mean(y_train)
        if train_prop_size == 1:
            continue
        elif sample_type == "gan":
            X_train, y_train = extend_gan_train(
                X_train,
                y_train,
                X_test,
                cat_cols,
                epochs=500,
                gen_x_times=train_prop_size,
            )
        elif sample_type == "sample_original":
            X_train, y_train = extend_from_original(
                X_train, y_train, X_test, cat_cols, gen_x_times=train_prop_size
            )
        y_train, y_test = y_train, y_test

        for encoders_tuple in encoders_list:
            print(
                f"\n{encoders_tuple}, {dataset_name}, train size {int(100 * train_prop_size)}%, "
                f"validation_type {validation_type}, sample_type {sample_type}"
            )

            time_start = time.time()

            # train models
            lgb_model = Model(
                cat_validation=validation_type,
                encoders_names=encoders_tuple,
                cat_cols=cat_cols,
            )
            train_score, val_score, avg_num_trees = lgb_model.fit(X_train, y_train)
            y_hat, test_features = lgb_model.predict(X_test)

            # check score
            test_score = roc_auc_score(y_test, y_hat)
            time_end = time.time()

            # write and save results
            results = {
                "dataset_name": dataset_name,
                "Encoder": encoders_tuple[0],
                "validation_type": validation_type,
                "sample_type": sample_type,
                "train_shape": X_train.shape[0],
                "test_shape": X_test.shape[0],
                "mean_target_before_sampling_train": mean_target_before_sampling_train,
                "mean_target_after_sampling_train": np.mean(y_train),
                "mean_target_test": np.mean(y_test),
                "num_cat_cols": len(cat_cols),
                "train_score": train_score,
                "val_score": val_score,
                "test_score": test_score,
                "time": time_end - time_start,
                "features_before_encoding": X_train.shape[1],
                "features_after_encoding": test_features,
                "avg_tress_number": avg_num_trees,
                "train_prop_size": train_prop_size,
            }
            save_exp_to_file(dic=results, path=f"./results/fit_predict_scores.txt")


if __name__ == "__main__":

    encoders_list = [("CatBoostEncoder",)]

    dataset_list = [
        "telecom",
        "adult",
        "employee",
        "mortgages",
        "poverty_A",
        "credit",
        "taxi",
    ]  # "kick","kdd_upselling"

    for dataset_name in tqdm(dataset_list):
        validation_type = "Single"
        execute_experiment(dataset_name, encoders_list, validation_type)
        execute_experiment(
            dataset_name, encoders_list, validation_type, sample_type="gan"
        )
        execute_experiment(
            dataset_name, encoders_list, validation_type, sample_type="sample_original"
        )
