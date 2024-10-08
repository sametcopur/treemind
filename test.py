import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from tree_explainer import Explainer
from tree_explainer.plots2 import (
    plot_bar,
    plot_values_points,
    plot_points,
    plot_interaction,
)

n_samples = 10000

np.random.seed(42)

feature_0 = np.random.normal(loc=0, scale=1, size=n_samples)
feature_1 = np.random.normal(loc=0, scale=1, size=n_samples)
feature_2 = np.random.normal(loc=0, scale=1, size=n_samples)
feature_3 = np.random.normal(loc=0, scale=1, size=n_samples)
feature_4 = np.random.normal(loc=0, scale=1, size=n_samples)

single_effects = (
    (np.where(feature_0 < 0, feature_0 * 3, 0))
    - (0.3 * np.tanh(feature_1))
    + (np.sin(feature_2) * 2)
    + (0.1 * np.exp(feature_3))
)

interaction_effects = (0.2 * feature_0 * feature_1) - (
    0.2 * (feature_2 * feature_3)
)

uclu = - ((feature_4 - feature_0) ** 2)

noise = np.random.normal(loc=0, scale=0.2, size=n_samples)

target = single_effects + interaction_effects + noise + uclu

target = np.where(target>= target.mean(), 1, 0 )

df = pd.DataFrame(
    {
        "feature_0": feature_0,
        "feature_1": feature_1,
        "feature_2": feature_2,
        "feature_3": feature_3,
        "feature_4": feature_4,
        "target": target,
    }
)


X = df.drop("target", axis=1).values
y = df["target"].values

train_data = lgb.Dataset(X, label=y)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "max_depth": 10,
    "verbosity": -1,
    
    "feature_fraction": 0.9,
}

num_round = 200
model = lgb.train(params, train_data, num_round)

import xgboost as xgb

clf = xgb.XGBClassifier().fit(X, y)

tree_explainer = Explainer()
tree_explainer(clf)
i = 7
x_xgb = xgb.DMatrix(X)
values, raw_score = tree_explainer.analyze_row(x_xgb, detailed=False)
plot_bar(values, raw_score)