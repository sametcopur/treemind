## Tree Explainer

The Tree Explainer library is designed to interpret ensemble tree models by breaking down individual trees and analyzing the model's predictions. It explains the model's decision-making process by calculating the expected value of the predictions at each decision point. The Tree Explainer library is fully integrated with `LightGBM` and `XGBoost`.

## Installation

To install the Tree Explainer library, ensure you are using Python 3.9 or higher. You can install the package via `pip`:

```bash
pip install tree_explainer
```

### Key Features

1. **Interaction Analysis:** Identifies complex relationships between features by analyzing how they work together to influence predictions.

2. **Feature Importance and Split Counting:** Determines how often individual features or feature pairs are used in the decision-making process, highlighting the most influential factors.

3. **Detailed Prediction Breakdown:** Analyzes individual predictions to show how features contribute step-by-step to the final output.

4. **Feature-specific Insights:** Provides statistical analysis on how features behave across different decision splits, including their typical ranges.

5. **High Performance:** Optimized with Cython for fast execution, even on large models and datasets.

6. **Advanced Visualization:** Offers user-friendly plots to visually explain the model's decision-making process and feature interactions. 

These features help users interpret ensemble models comprehensively, providing both quantitative insights and visual explanations.

### Tree Explainer Algorithm Overview

The Tree Explainer algorithm combines predictions from multiple decision trees by analyzing split points and calculating the average outcomes for different feature intervals. The process involves:

1. **Identifying Split Points:** The algorithm identifies distinct split points across all trees for a given feature. These split points determine the intervals where the predictions need to be calculated.

2. **Analyzing Intervals:** For each interval, the algorithm determines the possible leaf nodes in each tree that correspond to that range. These leaf nodes provide the raw prediction scores for the given feature values.

3. **Calculating Combinations of Raw Scores:** It computes all possible combinations of raw scores from the leaf nodes across the trees within the interval.

4. **Averaging the Results:** The algorithm calculates the average of the combined raw scores for each interval, yielding a final predicted outcome for that range.

By repeating this process for all feature intervals, the algorithm generates a comprehensive prediction by aggregating the results from all decision trees.

