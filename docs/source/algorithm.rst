.. _treemind_algorithm:

treemind Algorithm
==================

1. Prediction Phase
---------------------

- Each tree :math:`t \in \{1, 2, ..., T\}` operates independently during the prediction phase.
- Each tree produces predictions in the form :math:`f_t(x)`.
- The total model prediction is the sum of the predictions from all trees:

  .. math::

     F(x) = \sum_{t=1}^{T} f_t(x)

This assumption implies that each tree contributes independently to the final prediction, allowing us to analyze each tree separately when assessing feature contributions.


2. Leaf Statistics for Each Split Node
------------------------------------------

Let :math:`S_t` denote the set of trees in the ensemble.

- **For tree** :math:`t` **and leaf** :math:`i` **in the tree** :math:`t`:

  - :math:`L_{t,i}`: Leaf count—the number of data points falling into leaf :math:`i` of tree :math:`t`.
  - :math:`V_{t,i}`: Leaf value—the prediction output of leaf :math:`i` in tree :math:`t`.
  - :math:`D_t(x)`: Set of leaves in tree :math:`t` where **the** :math:`x` **node matches the analysis interval**.

**Establishing Intervals**

Before determining :math:`D_t(x)`, we traverse each tree to establish intervals for the analyzed feature. This process consists of the following steps:

1. **Tree Traversal to Identify Split Nodes**  
   Each tree is traversed to locate nodes where the analyzed feature is used for splitting. For each such node:
   
   - Record the split value associated with the analyzed feature.  
   - Collect these split values into a set :math:`S = \{s_1, s_2, ..., s_k\}`, ensuring no duplicates, where :math:`s_1 < s_2 < ... < s_k`.

2. **Defining Intervals**  
   Using the collected split values, the following intervals are constructed:
   
   - :math:`(-\infty, s_1]`  
   - :math:`(s_1, s_2]`  
   - :math:`(s_2, s_3]`  
   - ...  
   - :math:`(s_{k-1}, s_k]`  
   - :math:`(s_k, \infty)`  

**Matching Intervals**

When determining :math:`D_t(x)`, we use the intervals established above. A leaf is included in :math:`D_t(x)` if it satisfies one of the following conditions:

1. **Analyzed Feature Exists in the Path**  
   If the analyzed feature :math:`x` is part of the splitting path of the leaf, the feature split value must align with the interval being analyzed.  
   For example, if the analysis interval is :math:`(-\infty, s_1]` and the leaf contains a split node with :math:`x \leq s_k`, it is valid and included in :math:`D_t(x)`.

2. **Analyzed Feature is Absent from the Path**  
   If the analyzed feature :math:`x` is not part of the splitting path for the leaf, the leaf is still included in :math:`D_t(x)`. This is because such leaves represent general conditions that can apply to the analyzed feature.

3. Calculation of Expected Value
----------------------------------

**Expected Value for Each Tree:**

For a specific interval of feature :math:`x` (e.g., :math:`x \in \text{interval}`), the expected value of tree :math:`t` is calculated as:

.. math::

   E[f_t(x) \mid x \in \text{interval}] = \frac{\sum\limits_{i \in D_t(x)} L_{t,i} \cdot V_{t,i}}{\sum\limits_{i \in D_t(x)} L_{t,i}}

This formula computes the weighted average of leaf values, where weights are the leaf counts.

**Average Data Count Across Trees:**

.. math::

   AC[x \in \text{interval}] = \frac{1}{|S_t|} \sum_{t \in S_t} \sum_{i \in D_t(x)} L_{t,i}

**Overall Expected Value:**

The average expected value across all trees using feature :math:`x` is:

.. math::

   E[F(x) \mid x \in \text{interval}] = \sum_{t \in S_t} E[f_t(x) \mid x \in \text{interval}]

This represents the aggregated contribution of feature :math:`x \in \text{interval}` over all relevant trees.

4. Difference from Mean
--------------------------

To evaluate the contribution of a feature within a specific interval, the difference is computed between the expected model output conditioned on that interval (:math:`E[F(x) \mid x \in \text{interval}]`) and the overall expected model output (:math:`E[F(x)]`).

The overall expected model output (:math:`E[F(x)]`) is calculated as:

.. math::

   E[F(x)] = \frac{\sum_{\text{interval}} E[F(x) \mid x \in \text{interval}] \cdot AC[x \in \text{interval}]}{\sum_{\text{interval}} AC[x \in \text{interval}]}

The contribution of feature :math:`x` within a given interval is then:

.. math::

   \text{Contribution}(x \in \text{interval}) = E[F(x) \mid x \in \text{interval}] - E[F(x)]

This formula ensures that the model's behavior is correctly aggregated across all intervals when calculating the baseline (:math:`E[F(x)]`), allowing for an accurate assessment of the feature's interval-specific influence.

5. Feature Interactions
-------------------------

1. **Interval Determination**

   For each selected tree:
   
   - Identify split points for `feature_1` to determine its intervals.  
   - Identify split points for `feature_2` to determine its intervals.  

2. **Combined Interval Analysis**

   For each combination of intervals from `feature_1` and `feature_2`, calculate the expected model output:

   .. math::
      E[F(x) \mid x_1 \in \text{interval}_1, x_2 \in \text{interval}_2]

The forward steps remain consistent as described, and this approach can be extended to accommodate additional features.

6. Back Data Integration
--------------------------

The treemind algorithm allows for the integration of back data, which dynamically updates the leaf counts to reflect the new data while 
keeping the tree structure (splits and leaf values) unchanged.


When new data **back data** is provided, the leaf counts are recalculated as:

.. math::

   L'_{t,i} = \sum_{d \in B} I(d \text{ falls into leaf } i)

where:

- :math:`L'_{t,i}`: Updated leaf count for leaf :math:`i` in tree :math:`t`
- :math:`B`: Set of back data instances
- :math:`I(d \text{ falls into leaf } i)`: Indicator function (1 if instance :math:`d` falls into leaf :math:`i`, 0 otherwise)

7. Mathematical Limitations and Practical Considerations
--------------------------------------------------------

.. note::

   1. **Theoretical Foundation:**
      Although the algorithm produces desired results in practice, it lacks formal mathematical proof.

   2. **Interpretation of Results:**
      - The differences obtained by subtracting the final expected value are for interpretative purposes only
      - These values do not reflect the true differences but show correlation with actual values

   3. **Asymptotic Performance:**
      The algorithm's performance improves as both:
      
      - Number of estimators (trees) approaches infinity
      - Amount of data approaches infinity