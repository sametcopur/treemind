Getting Started with TreeMind
=============================

TreeMind is a powerful, flexible Python library for interpreting decision tree-based models with ease. It provides intuitive tools for analyzing feature importance, understanding feature interactions, and visualizing decision paths. Built for speed and functionality, TreeMind is optimized for high-performance analysis of complex tree ensembles like gradient boosting and random forests.

Installation
------------

Install TreeMind via pip:

.. code-block:: bash

   pip install treemind


Key Features
------------

1. **Feature Analysis:** Provides statistical analysis on how features behave across different decision splits.

2. **Interaction Analysis:**  Identifies complex relationships between features by analyzing how they work together to influence predictions. The algorithm can analyze interactions up to n features, depending on memory constraints and time limitations.

3. **High Performance:** Optimized with Cython for fast execution, even on large models and datasets.

4. **Advanced Visualization:** Offers user-friendly plots to visually explain the model's decision-making process and feature interactions. 

5. **Compatibility with Popular Frameworks:** Fully compatible with `XGBoost`, `LightGBM` and `CatBoost`, supporting regression and binary classification tasks.
