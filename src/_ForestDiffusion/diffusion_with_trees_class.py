import copy
import math
from functools import partial

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import delayed, Parallel
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from _ForestDiffusion.utils.diffusion import VPSDE, get_pc_sampler
from _ForestDiffusion.utils.utils_diffusion import build_data_xt, euler_solve


## Class for the flow-matching or diffusion model
# Categorical features should be numerical (rather than strings), make sure to use x = pd.factorize(x)[0] to make them as such
# Make sure to specific which features are categorical and which are integers
# Note: Binary features can be considered integers since they will be rounded to the nearest integer and then clipped
class ForestDiffusionModel():

    def __init__(self, X,
                 label_y=None,
                 # must be a categorical/binary variable; if provided will learn multiple models for each label y
                 n_t=50,  # number of noise level
                 model='xgboost',  # xgboost, random_forest, lgbm, catboost
                 diffusion_type='flow',  # vp, flow (flow is better, but only vp can be used for imputation)
                 max_depth=7, n_estimators=100, eta=0.3,  # xgboost hyperparameters
                 num_leaves=31,  # lgbm hyperparameters
                 duplicate_K=100,  # number of different noise sample per real data sample
                 bin_indexes=[],  # vector which indicates which column is binary
                 cat_indexes=[],  # vector which indicates which column is categorical (>=3 categories)
                 int_indexes=[],
                 # vector which indicates which column is an integer (ordinal variables such as number of cats in a box)
                 true_min_max_values=None,
                 # Vector of form [[min_x, min_y], [max_x, max_y]]; If  provided, we use these values as the min/max for each variables when using clipping
                 gpu_hist=False,  # using GPU or not
                 eps=1e-3,
                 beta_min=0.1,
                 beta_max=8,
                 n_jobs=-1,
                 # cpus used (feel free to limit it to something small, this will leave more cpus per model; for lgbm you have to use n_jobs=1, otherwise it will never finish)
                 seed=666):  # Duplicate the dataset for improved performance

        np.random.seed(seed)

        # Sanity check, must remove observations with only missing data
        obs_to_remove = np.isnan(X).all(axis=1)
        X = X[~obs_to_remove]
        if label_y is not None:
            label_y = label_y[~obs_to_remove]

        int_indexes = int_indexes + bin_indexes  # since we round those, we do not need to dummy-code the binary variables

        if true_min_max_values is not None:
            self.X_min = true_min_max_values[0]
            self.X_max = true_min_max_values[1]
        else:
            self.X_min = np.nanmin(X, axis=0, keepdims=1)
            self.X_max = np.nanmax(X, axis=0, keepdims=1)

        self.cat_indexes = cat_indexes
        self.int_indexes = int_indexes
        if len(self.cat_indexes) > 0:
            X, self.X_names_before, self.X_names_after = self.dummify(X)  # dummy-coding for categorical variables

        # min-max normalization, this applies to dummy-coding too to ensure that they become -1 or +1
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        X = self.scaler.fit_transform(X)

        X1 = X
        self.X1 = copy.deepcopy(X1)
        self.b, self.c = X1.shape
        self.n_t = n_t
        self.duplicate_K = duplicate_K
        self.model = model
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.num_leaves = num_leaves
        self.eta = eta
        self.gpu_hist = gpu_hist
        self.label_y = label_y
        self.n_jobs = n_jobs

        if model == 'random_forest' and np.sum(np.isnan(X1)) > 0:
            raise ValueError('The dataset must not contain missing data in order to use model=random_forest')

        assert diffusion_type == 'vp' or diffusion_type == 'flow'
        self.diffusion_type = diffusion_type
        self.sde = None
        self.eps = eps
        self.beta_min = beta_min
        self.beta_max = beta_max
        if diffusion_type == 'vp':
            self.sde = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=n_t)

        if duplicate_K > 1:  # we duplicate the data multiple times, so that X0 is k times bigger so we have more room to learn
            X1 = np.tile(X1, (duplicate_K, 1))

        X0 = np.random.normal(size=X1.shape)  # Noise data

        if self.label_y is not None:
            assert np.sum(np.isnan(
                self.label_y)) == 0  # cannot have missing values in the label (just make a special categorical for nan if you need)
            self.y_uniques, self.y_probs = np.unique(self.label_y, return_counts=True)
            self.y_probs = self.y_probs / np.sum(self.y_probs)
            self.mask_y = {}  # mask for which observations has a specific value of y
            for i in range(len(self.y_uniques)):
                self.mask_y[self.y_uniques[i]] = np.zeros(self.b, dtype=bool)
                self.mask_y[self.y_uniques[i]][self.label_y == self.y_uniques[i]] = True
                self.mask_y[self.y_uniques[i]] = np.tile(self.mask_y[self.y_uniques[i]], (duplicate_K))
        else:  # assuming a single unique label 0
            self.y_probs = np.array([1.0])
            self.y_uniques = np.array([0])
            self.mask_y = {}  # mask for which observations has a specific value of y
            self.mask_y[0] = np.ones(X1.shape[0], dtype=bool)

        # Make Datasets of interpolation
        X_train, y_train = build_data_xt(X0, X1, n_t=self.n_t, diffusion_type=self.diffusion_type, eps=self.eps,
                                         sde=self.sde)

        # Fit model(s)
        n_steps = n_t
        n_y = len(self.y_uniques)  # for each class train a seperate model

        if self.n_jobs == 1:
            self.regr = [[[None for k in range(self.c)] for i in range(n_steps)] for j in self.y_uniques]
            for i in range(n_steps):
                for j in range(len(self.y_uniques)):
                    for k in range(self.c):
                        self.regr[j][i][k] = self.train_parallel(
                            X_train.reshape(self.n_t, self.b * self.duplicate_K, self.c)[i][self.mask_y[j], :],
                            y_train.reshape(self.b * self.duplicate_K, self.c)[self.mask_y[j], k]
                        )
        else:
            self.regr = Parallel(n_jobs=self.n_jobs)(  # using all cpus
                delayed(self.train_parallel)(
                    X_train.reshape(self.n_t, self.b * self.duplicate_K, self.c)[i][self.mask_y[j], :],
                    y_train.reshape(self.b * self.duplicate_K, self.c)[self.mask_y[j], k]
                ) for i in range(n_steps) for j in self.y_uniques for k in range(self.c)
            )
            # Replace fits with doubly loops to make things easier
            self.regr_ = [[[None for k in range(self.c)] for i in range(n_steps)] for j in self.y_uniques]
            current_i = 0
            for i in range(n_steps):
                for j in range(len(self.y_uniques)):
                    for k in range(self.c):
                        self.regr_[j][i][k] = self.regr[current_i]
                        current_i += 1
            self.regr = self.regr_

    def train_parallel(self, X_train, y_train):

        if self.model == 'random_forest':
            out = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                        random_state=self.seed)
        elif self.model == 'lgbm':
            out = LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves, learning_rate=0.1,
                                random_state=self.seed, force_col_wise=True)
        elif self.model == 'catboost':
            raise NotImplemented("catboost usage has been disabled, please use 'random_forest', 'lgbm' or 'xgboost' "
                                 "instead")
        elif self.model == 'xgboost':
            out = xgb.XGBRegressor(n_estimators=self.n_estimators, objective='reg:squarederror', eta=self.eta,
                                   max_depth=self.max_depth,
                                   reg_lambda=0.0, subsample=1.0, seed=self.seed,
                                   tree_method='gpu_hist' if self.gpu_hist else 'hist',
                                   gpu_id=0 if self.gpu_hist else None)
        else:
            raise Exception("model value does not exists")

        y_no_miss = ~np.isnan(y_train)
        out.fit(X_train[y_no_miss, :], y_train[y_no_miss])

        return out

    def dummify(self, X):
        df = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])  # to Pandas
        df_names_before = df.columns
        for i in self.cat_indexes:
            df = pd.get_dummies(df, columns=[str(i)], prefix=str(i), dtype='float', drop_first=True)
        df_names_after = df.columns
        df = df.to_numpy()
        return df, df_names_before, df_names_after

    def unscale(self, X):
        if self.scaler is not None:  # unscale the min-max normalization
            X = self.scaler.inverse_transform(X)
        return X

    # Rounding for the categorical variables which are dummy-coded and then remove dummy-coding
    def clean_onehot_data(self, X):
        if len(self.cat_indexes) > 0:  # ex: [5, 3] and X_names_after [gender_a gender_b cartype_a cartype_b cartype_c]
            X_names_after = copy.deepcopy(self.X_names_after.to_numpy())
            prefixes = [x.split('_')[0] for x in self.X_names_after if
                        '_' in x]  # for all categorical variables, we have prefix ex: ['gender', 'gender']
            unique_prefixes = np.unique(prefixes)  # uniques prefixes
            for i in range(len(unique_prefixes)):
                cat_vars_indexes = [unique_prefixes[i] + '_' in my_name for my_name in self.X_names_after]
                cat_vars_indexes = np.where(cat_vars_indexes)[0]  # actual indexes
                cat_vars = X[:, cat_vars_indexes]  # [b, c_cat]
                # dummy variable, so third category is true if all dummies are 0
                cat_vars = np.concatenate((np.ones((cat_vars.shape[0], 1)) * 0.5, cat_vars), axis=1)
                # argmax of -1, -1, 0 is 0; so as long as they are below 0 we choose the implicit-final class
                max_index = np.argmax(cat_vars, axis=1)  # argmax across all the one-hot features (most likely category)
                X[:, cat_vars_indexes[0]] = max_index
                X_names_after[cat_vars_indexes[0]] = unique_prefixes[i]  # gender_a -> gender
            df = pd.DataFrame(X, columns=X_names_after)  # to Pandas
            df = df[self.X_names_before]  # remove all gender_b, gender_c and put everything in the right order
            X = df.to_numpy()
        return X

    # Unscale and clip to prevent going beyond min-max and also round of the integers
    def clip_extremes(self, X):
        if self.int_indexes is not None:
            for i in self.int_indexes:
                X[:, i] = np.round(X[:, i], decimals=0)
        small = (X < self.X_min).astype(float)
        X = small * self.X_min + (1 - small) * X
        big = (X > self.X_max).astype(float)
        X = big * self.X_max + (1 - big) * X
        return X

    # Return the score-fn or ode-flow output
    def my_model(self, t, y, mask_y=None):
        # y is [b*c]
        c = self.c
        b = y.shape[0] // c
        X = y.reshape(b, c)  # [b, c]

        # Output
        out = np.zeros(X.shape)  # [b, c]
        i = int(round(t * (self.n_t - 1)))
        for j, label in enumerate(self.y_uniques):
            if mask_y[label].sum() > 0:
                for k in range(self.c):
                    out[mask_y[label], k] = self.regr[j][i][k].predict(X[mask_y[label], :])

        if self.diffusion_type == 'vp':
            alpha_, sigma_ = self.sde.marginal_prob_coef(X, t)
            out = - out / sigma_
        out = out.reshape(-1)  # [b*c]
        return out

    # For imputation, we only give out and receive the missing values while ensuring consistency for the non-missing values
    # y0 is prior data, X_miss is real data
    def my_model_imputation(self, t, y, X_miss, sde=None, mask_y=None):

        y0 = np.random.normal(size=X_miss.shape)  # Noise data
        b, c = y0.shape

        if self.diffusion_type == 'vp':
            assert sde is not None
            mean, std = sde.marginal_prob(X_miss, t)
            X = mean + std * y0  # following the sde
        else:
            X = t * X_miss + (1 - t) * y0  # interpolation based on ground-truth for non-missing data
        mask_miss = np.isnan(X_miss)
        X[mask_miss] = y  # replace missing data by y(t)

        # Output
        out = np.zeros(X.shape)  # [b, c]
        i = int(round(t * (self.n_t - 1)))
        for j, label in enumerate(self.y_uniques):
            if mask_y[label].sum() > 0:
                for k in range(self.c):
                    out[mask_y[label], k] = self.regr[j][i][k].predict(X[mask_y[label], :])

        if self.diffusion_type == 'vp':
            alpha_, sigma_ = self.sde.marginal_prob_coef(X, t)
            out = - out / sigma_

        out = out[mask_miss]  # only return the missing data output
        out = out.reshape(-1)  # [-1]
        return out

    # Generate new data by solving the reverse ODE/SDE
    def generate(self, batch_size=None, n_t=None):

        # Generate prior noise
        y0 = np.random.normal(size=(self.b if batch_size is None else batch_size, self.c))

        # Generate random labels
        label_y = self.y_uniques[np.argmax(np.random.multinomial(1, self.y_probs, size=y0.shape[0]), axis=1)]
        mask_y = {}  # mask for which observations has a specific value of y
        for i in range(len(self.y_uniques)):
            mask_y[self.y_uniques[i]] = np.zeros(y0.shape[0], dtype=bool)
            mask_y[self.y_uniques[i]][label_y == self.y_uniques[i]] = True
        my_model = partial(self.my_model, mask_y=mask_y)

        if self.diffusion_type == 'vp':
            sde = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=self.n_t if n_t is None else n_t)
            ode_solved = get_pc_sampler(my_model, sde=sde, denoise=True, eps=self.eps)(y0.reshape(-1))
        else:
            ode_solved = euler_solve(my_model=my_model, y0=y0.reshape(-1),
                                     N=self.n_t if n_t is None else n_t)  # [t, b*c]
        solution = ode_solved.reshape(y0.shape[0], self.c)  # [b, c]
        solution = self.unscale(solution)
        solution = self.clean_onehot_data(solution)
        solution = self.clip_extremes(solution)

        # Concatenate y label if needed
        if self.label_y is not None:
            solution = np.concatenate((solution, np.expand_dims(label_y, axis=1)), axis=1)

        return solution

    # Impute missing data by solving the reverse ODE while keeping the non-missing data intact
    def impute(self, k=1, X=None, label_y=None, repaint=False, r=5, j=0.1, n_t=None):  # X is data with missing values
        assert self.diffusion_type != 'flow'  # cannot use with flow=matching

        if X is None:
            X = self.X1
        if label_y is None:
            label_y = self.label_y
        if n_t is None:
            n_t = self.n_t

        if self.diffusion_type == 'vp':
            sde = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=n_t)

        if label_y is None:  # single category 0
            mask_y = {}
            mask_y[0] = np.ones(X.shape[0], dtype=bool)
        else:
            mask_y = {}  # mask for which observations has a specific value of y
            for i in range(len(self.y_uniques)):
                mask_y[self.y_uniques[i]] = np.zeros(X.shape[0], dtype=bool)
                mask_y[self.y_uniques[i]][label_y == self.y_uniques[i]] = True

        my_model_imputation = partial(self.my_model_imputation, X_miss=X, sde=sde, mask_y=mask_y)

        for i in range(k):
            y0 = np.random.normal(size=X.shape)

            mask_miss = np.isnan(X)
            y0_miss = y0[mask_miss].reshape(-1)
            solution = copy.deepcopy(X)  # Solution start with dataset which contains some missing values
            if self.diffusion_type == 'vp':
                ode_solved = get_pc_sampler(my_model_imputation, sde=sde, denoise=True, repaint=repaint)(y0_miss, r=r,
                                                                                                         j=int(
                                                                                                             math.ceil(
                                                                                                                 j * n_t)))
                solution[mask_miss] = ode_solved  # replace missing values with imputed values
            solution = self.unscale(solution)
            solution = self.clean_onehot_data(solution)
            solution = self.clip_extremes(solution)
            # Concatenate y label if needed
            if self.label_y is not None:
                solution = np.concatenate((solution, np.expand_dims(label_y, axis=1)), axis=1)
            if i == 0:
                imputed_data = np.expand_dims(solution, axis=0)
            else:
                imputed_data = np.concatenate((imputed_data, np.expand_dims(solution, axis=0)), axis=0)
        return imputed_data[0] if k == 1 else imputed_data
