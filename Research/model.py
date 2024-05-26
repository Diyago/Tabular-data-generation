import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import warnings

from encoders import MultipleEncoder, DoubleValidationEncoderNumerical


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
        warnings.filterwarnings("ignore", category=UserWarning)
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
        print(f"Mean score train : {np.round(mean_score_train, 4)}")
        print(f"Mean score val : {np.round(mean_score_val, 4)}")
        return mean_score_train, mean_score_val, avg_num_trees

    def predict(self, X: pd.DataFrame, return_shape=True) -> np.array:
        """
        Making inference with trained models for input dataframe
        Args:
            X: input dataframe for inference
            return_shape: boolean return shape if True

        Returns: Predicted ranks and number of input features if return_shape is True

        """
        y_hat = np.zeros(X.shape[0])
        for encoder, model in zip(self.encoders_list, self.models_list):
            X_test = X.copy()
            X_test = encoder.transform(X_test)

            # check for OrdinalEncoder encoding
            for col in [col for col in X_test.columns if "OrdinalEncoder" in col]:
                X_test[col] = X_test[col].astype("category")

            unranked_preds = model.predict_proba(X_test)[:, 1]
            y_hat += rankdata(unranked_preds)
        if return_shape:
            return y_hat, X_test.shape[1]
        return y_hat
