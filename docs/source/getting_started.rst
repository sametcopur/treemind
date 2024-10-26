Getting Started with TreeMind
=============================

TreeMind is a powerful, flexible Python library for interpreting decision tree-based models with ease. It provides intuitive tools for analyzing feature importance, understanding feature interactions, and visualizing decision paths. Built for speed and functionality, TreeMind is optimized for high-performance analysis of complex tree ensembles like gradient boosting and random forests.

Installation
------------

Install TreeMind via pip:

.. code-block:: bash

   pip install treemind

Overview
--------

TreeMind simplifies model interpretation by analyzing and visualizing tree structures. With TreeMind, you can explore split points, evaluate feature importance, and calculate predictions based on specific feature intervals. The library supports weighted raw scores and interaction analysis, making it ideal for identifying nuanced patterns within your data.

Key Features
------------

1. **Interaction Analysis:** Identifies complex relationships between features by analyzing how they work together to influence predictions.

2. **Feature Importance and Split Counting:** Determines how often individual features or feature pairs are used in the decision-making process, highlighting the most influential factors.

3. **Detailed Prediction Breakdown:** Analyzes individual predictions to show how features contribute step-by-step to the final output.

4. **Feature-specific Insights:** Provides statistical analysis on how features behave across different decision splits, including their typical ranges.

5. **High Performance:** Optimized with Cython for fast execution, even on large models and datasets.

6. **Advanced Visualization:** Offers user-friendly plots to visually explain the model's decision-making process and feature interactions.

These features help users interpret ensemble models comprehensively, providing both quantitative insights and visual explanations.
