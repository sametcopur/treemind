
from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer

from treemind import Explainer
from treemind.plot import (
    bar_plot,
    range_plot,
    feature_plot,
    interaction_plot,
)

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

model = LGBMClassifier(verbose=-1)
model.fit(X, y)

explainer = Explainer()
explainer(model)

feature_df = explainer.analyze_feature(2)
feature_plot(feature_df)