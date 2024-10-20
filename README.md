## treemind 
The treemind library is designed to interpret ensemble tree models by breaking down individual trees and analyzing the model's predictions. It explains the model's decision-making process by calculating the expected value of the predictions at each decision point. The treemind library is fully integrated with `LightGBM` and `XGBoost`.

## Installation
To install `treemind`, use the following pip command:

```bash
pip install treemind
```

### Key Features

1. **Interaction Analysis:** Identifies complex relationships between features by analyzing how they work together to influence predictions.

2. **Feature Importance and Split Counting:** Determines how often individual features or feature pairs are used in the decision-making process, highlighting the most influential factors.

3. **Detailed Prediction Breakdown:** Analyzes individual predictions to show how features contribute step-by-step to the final output.

4. **Feature-specific Insights:** Provides statistical analysis on how features behave across different decision splits, including their typical ranges.

5. **High Performance:** Optimized with Cython for fast execution, even on large models and datasets.

6. **Advanced Visualization:** Offers user-friendly plots to visually explain the model's decision-making process and feature interactions. 

These features help users interpret ensemble models comprehensively, providing both quantitative insights and visual explanations.

### treemind Algorithm Overview

The treemind algorithm combines predictions from multiple decision trees by analyzing split points and calculating the average outcomes for different feature intervals. The process involves:

1. **Identifying Split Points:** The algorithm identifies distinct split points across all trees for a given feature. These split points determine the intervals where the predictions need to be calculated.

2. **Analyzing Intervals:** For each interval, the algorithm determines the possible leaf nodes in each tree that correspond to that range. These leaf nodes provide the raw prediction scores for the given feature values.

3. **Calculating Combinations of Raw Scores:** It computes all possible combinations of raw scores from the leaf nodes across the trees within the interval.

4. **Averaging the Results:** The algorithm calculates the average of the combined raw scores for each interval, yielding a final predicted outcome for that range.

By repeating this process for all feature intervals, the algorithm generates a comprehensive prediction by aggregating the results from all decision trees.

### Usage

To use `treemind`, you need to initialize the `treemind` class with your specific parameters and fit it to your data. Here's a basic example:


```python
# Import necessary libraries for data handling, model training, and explanation
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from lightgbm import LGBMClassifier

# Import the Explainer class and plotting utility from the TreeMind library
from treemind import Explainer
from treemind.plot import plot_feature

# Set a random state for reproducibility
random_state = 42

# Load the Iris dataset (alternatively, you can load a different dataset like breast cancer)
# The data is split into features (X) and target variable (y)
X, y = load_iris(return_X_y=True)

# Split the dataset into training and testing sets
# The test size is set to 20%, meaning 80% of the data will be used for training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

# Initialize the LightGBM classifier
clf = LGBMClassifier()

# Train the classifier using the training data
clf.fit(X_train, y_train)

# Create an instance of the Explainer class from TreeMind
explainer = Explainer()

# Fit the explainer to the trained classifier
explainer(clf)

# Analyze a specific feature by its index (e.g., feature index 2)
# This step generates a detailed report on how this feature contributes to the model's predictions
feature_df = explainer.analyze_feature(2)

# Plot the analysis results for the selected feature
# This visualization helps understand the impact of the feature across different splits
plot_feature(feature_df)
```
### Documentation
For more detailed information about the API and advanced usage, please refer to the full  [documentation](https://treemind.readthedocs.io/en/latest/).

### Contributing
Contributions are welcome! If you'd like to improve `treemind` or suggest new features, feel free to fork the repository and submit a pull request.

### License
`treemind` is released under the BSD 3-Clause License. See the LICENSE file for more details.