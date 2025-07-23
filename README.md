
# treemind

`treemind` is a high-performance library for interpreting tree-based models. It supports regression, binary and multiclass classification, and handles both numerical and categorical features. By analyzing split intervals and feature interactions, `treemind` helps you understand which features drive predictions and how they interact making it ideal for model explanation, debugging, and auditing.


> A formal research paper detailing the theoretical foundation of `treemind` is forthcoming.

---

## Installation

Install `treemind` via pip:

```bash
pip install treemind
```

---

## Key Features

* **Feature Analysis**
  Quantifies how individual features influence predictions across specific decision boundaries.

* **Interaction Detection**
  Detects and visualizes interaction effects between two or more features at any order `n`, constrained by memory and time.

* **Optimized Performance**
  Fast even on deep models thanks to efficient Cython-backed core.

* **Rich Visualizations**
  Interactive and static plots to visualize importance, split intervals, and interaction strength.

* **Broad Model Support**
  Compatible with `xgboost`, `lightgbm`, `catboost`, `sklearn`, and `perpetual`. Works with regression, binary, and multiclass tasks. Supports categorical features.

---

## Algorithm & Performance

The `treemind` algorithm analyzes how often features and their combinations appear in decision paths, then summarizes their behavior over split intervals.

* [Algorithm Overview](https://treemind.readthedocs.io/en/latest/algorithm.html)
* [Performance Benchmarks](https://treemind.readthedocs.io/en/latest/experiments/experiment_main.html)

---

### Quickstart Example

This walkthrough shows how to use `treemind.Explainer` with a LightGBM model trained on the Breast Cancer dataset.

```python
from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer

from treemind import Explainer
from treemind.plot import (
    feature_plot,
    interaction_plot,
    interaction_scatter_plot,
    importance_plot,
)

# Load sample data
X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# Train a model
model = LGBMClassifier(verbose=-1)
model.fit(X, y)

# Create an explainer
explainer = Explainer(model)
```

---

### Count Feature Appearances

To see how often each feature (or feature pair) appears in the decision trees:

```python
explainer.count_node(degree=1)  # Individual feature usage
```

```text
| column_index | count |
|--------------|-------|
| 21           | 1739  |
| 27           | 1469  |
```

```python
explainer.count_node(degree=2)  # Pairwise feature usage
```

```text
| column1_index | column2_index | count |
|---------------|---------------|-------|
| 21            | 22            | 927   |
| 21            | 23            | 876   |
```

---

### One-Dimensional Feature Analysis

Analyze how a single feature influences the model:

```python
result1_d = explainer.explain(degree=1)
```

Inspect a specific feature (e.g., feature 21):

```python
result1_d[21]
```

```text
| worst_texture_lb | worst_texture_ub | value     | std      | count  |
|------------------|------------------|-----------|----------|--------|
| -inf             | 18.460           | 3.185128  | 8.479232 | 402.24 |
| 18.460           | 19.300           | 3.160656  | 8.519873 | 402.39 |
```

#### Feature Visualization

```python
feature_plot(result1_d, 21)
```

<p align="center">
  <img src="/docs/source/_static/api/feature_plot.png" alt="Feature Plot" width="80%">
</p>

#### Feature Importance

```python
result1_d.importance()
```

```text
| feature_0            | importance |
|----------------------|------------|
| worst_concave_points | 2.326004   |
| worst_perimeter      | 2.245493   |
```

```python
importance_plot(result1_d)
```

<p align="center">
  <img src="/docs/source/_static/api/importance_plot.png" alt="Feature Importance" width="80%">
</p>

---

### Two-Dimensional Interaction Analysis

Evaluate how two features interact to influence predictions:

```python
result2_d = explainer.explain(degree=2)
result2_d[21, 22]
```

```text
| worst_texture_lb | worst_texture_ub | worst_concave_points_lb | worst_concave_points_ub | value    | std      | count  |
|------------------|------------------|--------------------------|--------------------------|----------|----------|--------|
| -inf             | 18.46            | -inf                     | 0.058860                 | 4.929324 | 7.679424 | 355.40 |
```

#### Interaction Importance

```python
result2_d.importance()
```

```text
| feature_0         | feature_1            | importance |
|------------------|----------------------|------------|
| worst_perimeter  | worst_area           | 2.728454   |
| worst_texture    | worst_concave_points | 2.439605   |
```

```python
importance_plot(result2_d)
```

<p align="center">
  <img src="/docs/source/_static/api/importance_plot2d.png" alt="2D Importance" width="80%">
</p>

#### Interaction Plots

```python
interaction_plot(result2_d, (21, 22))
```

<p align="center">
  <img src="/docs/source/_static/api/interaction_plot.png" alt="Interaction Plot" width="80%">
</p>

```python
interaction_scatter_plot(X, result2_d, (21, 22))
```

<p align="center">
  <img src="/docs/source/_static/api/interaction_scatter_plot.png" alt="Interaction Scatter" width="80%">
</p>

---



## Contributing

Contributions are welcome! If you'd like to improve `treemind` or suggest new features, feel free to fork the repository and submit a pull request.

---

## License

`treemind` is released under the MIT License. See the [LICENSE](./LICENSE) file for details.
