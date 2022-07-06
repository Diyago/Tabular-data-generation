import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from tabgan.encoders import MultipleEncoder, DoubleValidationEncoderNumerical


class AdversarialModel:
    def __init__(
            self,
            cat_validation="Single",
            encoders_names=("OrdinalEncoder",),
            cat_cols=None,
            model_validation=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            model_params=None,
    ):
        '''
        Class for fit predicting tabular models, mostly - boostings. Several encoders for categorical features are
        supported

        Args:
            cat_validation: categorical type of validation, examples: "None", "Single" and "Double"
            encoders_names: different categorical encoders from category_encoders library, example CatBoostEncoder
            cat_cols: list of categorical columns
            model_validation: model training cross validation type from sklearn.model_selection,
            example StratifiedKFold(5)
            model_params: model training hyperparameters
        '''
        self.cat_validation = cat_validation
        self.encoders_names = encoders_names
        self.cat_cols = cat_cols
        self.model_validation = model_validation
        self.model_params = model_params

    def adversarial_test(self, left_df, right_df):
        """
        Trains adversarial model to distinguish train from test
        :param left_df:  dataframe
        :param right_df: dataframe
        :return: trained model
        """
        # sample to shuffle the data
        left_df = left_df.copy().sample(frac=1).reset_index(drop=True)
        right_df = right_df.copy().sample(frac=1).reset_index(drop=True)

        left_df = left_df.head(right_df.shape[0])
        right_df = right_df.head(left_df.shape[0])

        left_df["gt"] = 0
        right_df["gt"] = 1

        concated = pd.concat([left_df, right_df])
        lgb_model = Model(
            cat_validation=self.cat_validation,
            encoders_names=self.encoders_names,
            cat_cols=self.cat_cols,
            model_validation=self.model_validation,
            model_params=self.model_params,
        )
        train_score, val_score, avg_num_trees = lgb_model.fit(
            concated.drop("gt", axis=1), concated["gt"]
        )
        self.metrics = {"train_score": train_score,
                        "val_score": val_score,
                        "avg_num_trees": avg_num_trees}
        self.trained_model = lgb_model


class Model:
    def __init__(
            self,
            cat_validation="None",
            encoders_names=None,
            cat_cols=None,
            model_validation=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            model_params=None,
    ):
        '''
        Class for fit predicting tabular models, mostly - boostings. Several encoders for categorical features are supported

        Args:
            cat_validation: categorical type of validation, examples: "None", "Single" and "Double"
            encoders_names: different categorical encoders from category_encoders library, example CatBoostEncoder
            cat_cols: list of categorical columns
            model_validation: model training cross validation type from sklearn.model_selection, example StratifiedKFold(5)
            model_params: model training hyperparameters
        '''
        self.cat_validation = cat_validation
        self.encoders_names = encoders_names
        self.cat_cols = cat_cols
        self.model_validation = model_validation

        if model_params is None:
            self.model_params = {
                "metrics": "AUC",
                "n_estimators": 5000,
                "learning_rate": 0.04,
                "random_state": 42,
            }
        else:
            self.model_params = model_params

        self.encoders_list = []
        self.models_list = []
        self.scores_list_train = []
        self.scores_list_val = []
        self.models_trees = []

    def fit(self, X: pd.DataFrame, y: np.array) -> tuple:
        """
        Fits model with speficified in init params
        Args:
            X: Input training dataframe
            y: Target for X

        Returns:
            mean_score_train, mean_score_val, avg_num_trees
        """
        # process cat cols
        if self.cat_validation == "None":
            encoder = MultipleEncoder(
                cols=self.cat_cols, encoders_names_tuple=self.encoders_names
            )
            X = encoder.fit_transform(X, y)

        for n_fold, (train_idx, val_idx) in enumerate(
                self.model_validation.split(X, y)
        ):
            X_train, X_val = (
                X.iloc[train_idx].reset_index(drop=True),
                X.iloc[val_idx].reset_index(drop=True),
            )
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            if self.cat_cols is not None:
                if self.cat_validation == "Single":
                    encoder = MultipleEncoder(
                        cols=self.cat_cols, encoders_names_tuple=self.encoders_names
                    )
                    X_train = encoder.fit_transform(X_train, y_train)
                    X_val = encoder.transform(X_val)
                if self.cat_validation == "Double":
                    encoder = DoubleValidationEncoderNumerical(
                        cols=self.cat_cols, encoders_names_tuple=self.encoders_names
                    )
                    X_train = encoder.fit_transform(X_train, y_train)
                    X_val = encoder.transform(X_val)
                self.encoders_list.append(encoder)

                # check for OrdinalEncoder encoding
                for col in [col for col in X_train.columns if "OrdinalEncoder" in col]:
                    X_train[col] = X_train[col].astype("category")
                    X_val[col] = X_val[col].astype("category")

            # fit model
            model = LGBMClassifier(**self.model_params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False,
            )
            self.models_trees.append(model.best_iteration_)
            self.models_list.append(model)

            y_hat = model.predict_proba(X_train)[:, 1]
            score_train = roc_auc_score(y_train, y_hat)
            self.scores_list_train.append(score_train)
            y_hat = model.predict_proba(X_val)[:, 1]
            score_val = roc_auc_score(y_val, y_hat)
            self.scores_list_val.append(score_val)

        mean_score_train = np.mean(self.scores_list_train)
        mean_score_val = np.mean(self.scores_list_val)
        avg_num_trees = int(np.mean(self.models_trees))

        return mean_score_train, mean_score_val, avg_num_trees

    def predict(self, X: pd.DataFrame) -> np.array:
        """
        Making inference with trained models for input dataframe
        Args:
            X: input dataframe for inference

        Returns: Predicted ranks

        """
        y_hat = np.zeros(X.shape[0])
        if self.encoders_list is not None and self.encoders_list != []:
            for encoder, model in zip(self.encoders_list, self.models_list):
                X_test = X.copy()
                X_test = encoder.transform(X_test)

                # check for OrdinalEncoder encoding
                for col in [col for col in X_test.columns if "OrdinalEncoder" in col]:
                    X_test[col] = X_test[col].astype("category")

                unranked_preds = model.predict_proba(X_test)[:, 1]
                y_hat += rankdata(unranked_preds)
        else:
            for model in self.models_list:
                X_test = X.copy()

                unranked_preds = model.predict_proba(X_test)[:, 1]
                y_hat += rankdata(unranked_preds)
        return y_hat
