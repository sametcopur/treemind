Getting Started with treemind
=============================

``treemind`` is a model interpretation library designed for analyzing decision tree-based models, including both standalone decision trees and ensemble methods like gradient boosting.  
It helps you understand how features influence predictions, both individually and in combination, by analyzing decision boundaries and feature intervals.

Installation
------------

Install ``treemind`` via pip:

.. code-block:: bash

   pip install treemind

Key Features
------------

1. **Feature Analysis:**  
   Provides detailed statistics on how individual features impact model predictions across various split intervals.

2. **Interaction Analysis:**  
   Reveals complex relationships between features by analyzing multi-way interactions. Supports interactions up to *n* features, depending on available memory and time.

3. **Broad Model and Task Support:**  
   Compatible with ``xgboost``,  ``lightgbm``, ``catboost``, ``perpetual``, and ``sklearn``, and supports regression, binary classification, **multiclass classification**, and **categorical features**.

4. **High Performance:**  
   Optimized with Cython for fast computation, even on large models and datasets.

5. **Advanced Visualization:**  
   Includes intuitive visual tools to explore feature importance, decision intervals, and interaction effects for better model interpretability.
