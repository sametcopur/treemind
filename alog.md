# Mathematical Analysis of Feature Interval Expected Outputs in Gradient Boosting Models

**Abstract:**  
Understanding the influence of individual features on the predictions of gradient boosting models is essential for interpretability and feature importance analysis. This paper presents a mathematical formulation of an algorithm that computes the expected model output over specific intervals of a feature. By interpreting these computations as conditional expectations, we provide insights into the feature's effect on the model's predictions. The algorithm leverages the structure of the ensemble trees and the distribution of the training data, culminating in an adjusted expected output for each interval.

---

## 1. Introduction

Gradient boosting is a powerful ensemble learning technique that builds predictive models by sequentially adding decision trees to minimize a specified loss function. While these models achieve high predictive performance, they often lack interpretability due to their complexity. Interpreting the influence of individual features on model predictions is crucial for understanding model behavior, diagnosing issues, and informing decision-making processes.

This paper introduces a mathematical framework for an algorithm designed to compute the expected output of a gradient boosting model over intervals of a specific feature. The algorithm calculates the conditional expectation of the model's prediction given that the feature falls within certain intervals, effectively quantifying the feature's influence. By aggregating these interval expectations and adjusting them relative to the overall expected value, we derive meaningful insights into how variations in the feature affect the model's output.

## 2. Algorithm Description

### 2.1 Notations and Definitions

- \( T \): Total number of trees in the ensemble.
- \( x \): Feature of interest.
- \( S \): Set of all split points for feature \( x \) across all trees.
- \( \mathcal{I} \): Set of intervals formed from \( S \), partitioning the domain of \( x \).
- \( I \in \mathcal{I} \): An interval in \( \mathcal{I} \).
- \( E[I] \): Expected model output when \( x \in I \).
- \( w[I] \): Weight associated with interval \( I \), representing the average count of training instances.
- \( \hat{y}_t(x) \): Prediction of tree \( t \) for input \( x \).
- \( L_t[I] \): Set of leaves in tree \( t \) reachable when \( x \in I \).
- \( v_{l} \): Output value (raw score) of leaf \( l \).
- \( c_{l} \): Number of training instances in leaf \( l \).

### 2.2 Algorithm Steps

#### Step 1: Extract Split Points and Construct Intervals

- **Collect Split Points:** Extract all unique split points \( S \) for feature \( x \) from every tree in the ensemble.
- **Construct Intervals:** Use the split points \( S \) to partition the domain of \( x \) into a set of intervals \( \mathcal{I} \).

#### Step 2: Compute Expected Output for Each Interval

For each interval \( I \in \mathcal{I} \):

1. **Initialize Variables:**

   \[
   E[I] = 0, \quad w[I] = 0, \quad \text{tree\_count} = 0
   \]

2. **For Each Tree \( t = 1, 2, \dots, T \):**

   - **Identify Relevant Leaves:**

     - Determine the leaves \( L_t[I] \) in tree \( t \) that are reachable when \( x \in I \).
     - These are the leaves consistent with \( x \) falling within interval \( I \).

   - **Compute Tree-Level Expected Output:**

     - **Tree Sum:**

       \[
       \text{tree\_sum}_t[I] = \sum_{l \in L_t[I]} v_{l} \cdot c_{l}
       \]

     - **Total Count:**

       \[
       \text{count}_t[I] = \sum_{l \in L_t[I]} c_{l}
       \]

     - **Number of Leaves:**

       \[
       \text{num\_leaves}_t[I] = |L_t[I]|
       \]

   - **Aggregate Tree Contributions:**

     - If \( \text{count}_t[I] > 0 \):

       \[
       \text{tree\_count} += 1
       \]

       \[
       w[I] += \frac{\text{count}_t[I]}{\text{num\_leaves}_t[I]}
       \]

       \[
       E[I] += \frac{\text{tree\_sum}_t[I]}{\text{count}_t[I]}
       \]

#### Step 3: Compute Overall Expected Feature Value

After processing all intervals:

1. **Compute Overall Expected Value \( E[\text{feature}] \):**

   \[
   E[\text{feature}] = \frac{\sum_{I \in \mathcal{I}} E[I] \cdot w[I]}{\sum_{I \in \mathcal{I}} w[I]}
   \]

2. **Adjust Interval Expected Outputs:**

   - For each interval \( I \):

     \[
     E_{\text{adjusted}}[I] = E[I] - E[\text{feature}]
     \]

## 3. Mathematical Interpretation

### 3.1 Conditional Expectation

The expected output for each interval \( E[I] \) can be interpreted as the conditional expectation of the model's prediction given that the feature \( x \) falls within interval \( I \):

\[
E[I] = \mathbb{E}[\hat{y}(x) \mid x \in I]
\]

where \( \hat{y}(x) = \sum_{t=1}^{T} \hat{y}_t(x) \) is the ensemble model's prediction for input \( x \).

### 3.2 Overall Expected Feature Value

The overall expected feature value \( E[\text{feature}] \) represents the expected model prediction across all possible values of \( x \), weighted by the distribution of training instances:

\[
E[\text{feature}] = \mathbb{E}[\hat{y}(x)]
\]

This value serves as a baseline for comparison against interval-specific expectations.

### 3.3 Adjusted Interval Outputs

The adjusted interval outputs \( E_{\text{adjusted}}[I] \) quantify the deviation of each interval's expected output from the overall expected value:

\[
E_{\text{adjusted}}[I] = \mathbb{E}[\hat{y}(x) \mid x \in I] - \mathbb{E}[\hat{y}(x)]
\]

This adjustment centers the interval expectations, highlighting the specific effect of the feature within each interval.

## 4. Interpretation and Implications

### 4.1 Feature Effect Quantification

By computing \( E_{\text{adjusted}}[I] \) for each interval, we obtain a detailed view of how changes in the feature \( x \) influence the model's predictions. Positive values of \( E_{\text{adjusted}}[I] \) indicate intervals where the feature contributes to an increase in the expected output, while negative values indicate a decrease.

### 4.2 Incorporation of Training Data Distribution

The weights \( w[I] \) reflect the average number of training instances associated with each interval. By weighting the interval expectations accordingly, the algorithm accounts for the empirical distribution of the feature values in the training data, ensuring that intervals with more data have a proportionally larger impact on the overall expectation.

### 4.3 Model Interpretability

This approach enhances model interpretability by:

- Providing a granular analysis of the feature's effect across different intervals.
- Highlighting intervals where the feature has a significant positive or negative impact.
- Allowing for the creation of partial dependence plots that visualize the relationship between the feature and the model's predictions.

## 5. Mathematical Justification

### 5.1 Law of Total Expectation

The computation of \( E[\text{feature}] \) aligns with the law of total expectation:

\[
E[\text{feature}] = \sum_{I \in \mathcal{I}} \mathbb{E}[\hat{y}(x) \mid x \in I] \cdot P(x \in I)
\]

In our algorithm, \( w[I] \) serves as a proxy for \( P(x \in I) \), as it represents the relative frequency of training instances within each interval.

### 5.2 Conditional Probability Approximation

The weights are computed using:

\[
w[I] = \sum_{t=1}^{T} \frac{\text{count}_t[I]}{\text{num\_leaves}_t[I]}
\]

This approximates the probability \( P(x \in I) \) by considering the proportion of training instances that fall within the interval \( I \) across all trees.

## 6. Conclusion

We have presented a mathematical formulation of an algorithm that computes the expected output of a gradient boosting model over specific intervals of a feature. By interpreting these computations as conditional expectations and adjusting them relative to the overall expected value, we gain valuable insights into the feature's influence on the model's predictions. This method enhances the interpretability of gradient boosting models and provides a foundation for further analysis of feature effects.

## 7. References

- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.

---

**Appendix: Summary of Key Equations**

1. **Interval Expected Output:**

   \[
   E[I] = \sum_{t=1}^{T} \frac{\sum\limits_{l \in L_t[I]} v_{l} \cdot c_{l}}{\sum\limits_{l \in L_t[I]} c_{l}}
   \]

2. **Overall Expected Feature Value:**

   \[
   E[\text{feature}] = \frac{\sum\limits_{I \in \mathcal{I}} E[I] \cdot w[I]}{\sum\limits_{I \in \mathcal{I}} w[I]}
   \]

3. **Adjusted Interval Outputs:**

   \[
   E_{\text{adjusted}}[I] = E[I] - E[\text{feature}]
   \]

**Note:** This mathematical framework provides a structured approach to dissect the impact of individual features within gradient boosting models. By focusing on the conditional expectations and leveraging the distribution of training data, the algorithm offers a robust method for feature effect analysis.
