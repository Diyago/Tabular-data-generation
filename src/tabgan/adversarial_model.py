import pandas as pd
from sklearn.model_selection import StratifiedKFold



class AdversarialModel:
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

    def adversarial_test(left_df, right_df, cat_cols):
        """
        Trains adversarial model to distinguish train from test
        :param left_df:  dataframe
        :param right_df: dataframe
        :param cat_cols: List of categorical columns
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
            cat_validation="Single",
            encoders_names=("OrdinalEncoder",),
            cat_cols=cat_cols,
            model_validation=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            model_params={
                "metrics": "AUC",
                "max_depth": 2,
                "max_bin": 100,
                "n_estimators": 500,
                "learning_rate": 0.02,
                "random_state": 42,
            },
        )
        train_score, val_score, avg_num_trees = lgb_model.fit(
            concated.drop("gt", axis=1), concated["gt"]
        )

        return lgb_model