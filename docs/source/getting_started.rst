Getting Started with treemind
=============================

treemind is designed for analyzing gradient boosting models. It simplifies understanding how features influence predictions
within specific intervals and provides powerful tools for analyzing individual features and their interactions.

Installation
------------

Install treemind via pip:

.. code-block:: bash

   pip install treemind


Key Features
------------

1. **Feature Analysis:** Provides statistical analysis on how features behave across different decision splits.

2. **Interaction Analysis:**  Identifies complex relationships between features by analyzing how they work together to influence predictions. The algorithm can analyze interactions up to n features, depending on memory constraints and time limitations.

3. **High Performance:** Optimized with Cython for fast execution, even on large models and datasets.

4. **Advanced Visualization:** Offers user-friendly plots to visually explain the model's decision-making process and feature interactions. 

5. **Compatibility with Popular Frameworks:** Fully compatible with ``xgboost``, ``lightgbm`` and ``catboost``, supporting regression and binary classification tasks.
