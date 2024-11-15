.. _treemind_algorithm:

treemind Algorithm
==================

This document provides a comprehensive explanation of the treemind algorithm, detailing its theoretical foundation and practical application through an example.

Algorithm Explanation
---------------------

1. Independence of Trees During Prediction Phase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Assumption of Independence:**

- Each tree :math:`t \in \{1, 2, ..., T\}` operates independently during the prediction phase.
- Each tree produces predictions in the form :math:`f_t(x)`.
- The total model prediction is the sum of the predictions from all trees:

  .. math::

     F(x) = \sum_{t=1}^{T} f_t(x)

This assumption implies that each tree contributes independently to the final prediction, allowing us to analyze each tree separately when assessing feature contributions.

2. Selection of Trees Where the Feature is Used as a Split Node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Set of Trees Utilizing the Feature:**

  .. math::

     S_x = \{ t \mid \text{Feature } x \text{ is used as a split node in tree } t \}

- **Total number of trees:** :math:`T`
- **Number of trees using feature** :math:`x`: :math:`|S_x| = m \leq T`

To analyze the effect of feature :math:`x`, we focus on the subset of trees where :math:`x` is used in the splitting criteria.

3. Leaf Statistics for Each Split Node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **For tree** :math:`t` **and leaf** :math:`i`:

  - :math:`L_{t,i}`: Leaf count—the number of data points falling into leaf :math:`i` of tree :math:`t`.
  - :math:`V_{t,i}`: Leaf value—the prediction output of leaf :math:`i` in tree :math:`t`.
  - :math:`D_t`: Set of leaves in tree :math:`t` where feature :math:`x` is used in the decision path **and the leaf's interval matches the analysis interval**.

**Matching Intervals:**

When determining :math:`D_t`, we must ensure that the intervals of feature :math:`x` represented by the leaves align with the interval we are analyzing. For instance, if the split points in the tree for feature :math:`x` are at 15, 18, and 20, this creates intervals:

- :math:`(-\infty, 15]`
- :math:`(15, 18]`
- :math:`(18, 20]`
- :math:`(20, \infty)`

We include a leaf in :math:`D_t` if its interval overlaps with the analysis interval. This careful matching ensures that we only consider leaves where feature :math:`x` directly influences the prediction within the specified range.

4. Calculation of Expected Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Expected Value for Each Tree:**

For a specific interval of feature :math:`x` (e.g., :math:`x \in \text{interval}`), the expected value of tree :math:`t` is calculated as:

.. math::

   E[f_t(x) \mid x \in \text{interval}] = \frac{\sum\limits_{i \in D_t} L_{t,i} \cdot V_{t,i}}{\sum\limits_{i \in D_t} L_{t,i}}

This formula computes the weighted average of leaf values, where weights are the leaf counts. **Only leaves where the interval of feature** :math:`x` **matches the analysis interval are included in the calculation.**

**Average Data Count Across Trees:**

.. math::

   AC[x \in \text{interval}] = \frac{1}{|S_x|} \sum_{t \in S_x} \left( \sum_{i \in D_t} L_{t,i} \right)

**Overall Expected Value:**

The average expected value across all trees using feature :math:`x` is:

.. math::

   E[F(x) \mid x \in \text{interval}] = \frac{\sum_{t \in S_x} E[f_t(x) \mid x \in \text{interval}] \cdot \left( \sum_{i \in D_t} L_{t,i} \right)}{\sum_{t \in S_x} \left( \sum_{i \in D_t} L_{t,i} \right)}

This represents the aggregated contribution of feature :math:`x` over all relevant trees, considering only the leaves that correspond to the specified interval.

Difference from Mean
~~~~~~~~~~~~~~~~~~~~

To evaluate the contribution of a feature within a specific interval, the difference is computed between the expected model output conditioned on that interval (:math:`E[F(x) \mid x \in \text{interval}]`) and the overall expected model output (:math:`E[F(x)]`).

The overall expected model output (:math:`E[F(x)]`) is calculated as:

.. math::

   E[F(x)] = \frac{\sum_{\text{interval}} E[F(x) \mid x \in \text{interval}] \cdot AC[x \in \text{interval}]}{\sum_{\text{interval}} AC[x \in \text{interval}]}

The contribution of feature :math:`x` within a given interval is then:

.. math::

   \text{Contribution}(x \in \text{interval}) = E[F(x) \mid x \in \text{interval}] - E[F(x)]

This formula ensures that the model's behavior is correctly aggregated across all intervals when calculating the baseline (:math:`E[F(x)]`), allowing for an accurate assessment of the feature's interval-specific influence.

6. Feature Interactions
~~~~~~~~~~~~~~~~~~~~~~~

1. **Tree Selection**
   - Filter decision trees to identify those that utilize both `feature_1` and `feature_2` as split nodes.
   - Within the selected trees, include only branches where both features are part of the decision path.

2. **Interval Determination**
   For each selected tree:
   
   a. Identify split points for `feature_1` to determine its intervals.  
   b. Identify split points for `feature_2` to determine its intervals.  

3. **Combined Interval Analysis**
   For each combination of intervals from `feature_1` and `feature_2`, calculate the expected model output:

   .. math::
      E[F(x) \mid x_1 \in \text{interval}_1, x_2 \in \text{interval}_2]

   This step quantifies how specific ranges of `feature_1` and `feature_2` interact to influence predictions.

4. **Leaf Selection**
   Narrow down to leaves that satisfy the following conditions:
   
   - The interval for `feature_1` corresponds to the target interval (`interval_1`).
   - The interval for `feature_2` corresponds to the target interval (`interval_2`).

By combining the above steps, this approach facilitates an in-depth understanding of feature interactions and their contributions to the model's predictions.



Example Application
-------------------


To illustrate the treemind algorithm, we will expand the previous example by including counts for each leaf node. This will allow us to calculate average data counts as specified in the algorithm.

**Tree 1:**

.. code-block:: none

   |--- feature_2 <= 1.5
   |   |--- raw_score: 1.25
   |   |--- leaf_count: 50
   |--- feature_2 > 1.5
   |   |--- feature_1 <= 2.5
   |   |   |--- raw_score: 1.57
   |   |   |--- leaf_count: 30
   |   |--- feature_1 > 2.5
   |   |   |--- raw_score: 2.10
   |   |   |--- leaf_count: 20

**Tree 2:**

.. code-block:: none

   |--- feature_2 <= 3.0
   |   |--- feature_1 <= 1.0
   |   |   |--- raw_score: 0.12
   |   |   |--- leaf_count: 40
   |   |--- feature_1 > 1.0
   |   |   |--- raw_score: 0.30
   |   |   |--- leaf_count: 35
   |--- feature_2 > 3.0
   |   |--- raw_score: 0.50
   |   |--- leaf_count: 25

Calculations for Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will calculate the expected values and average data counts for the following intervals of **feature_2**:

The split points for **feature_2** across both trees are identified as:

- Tree 1: **1.5**
- Tree 2: **3.0**

Based on these split points, the intervals for **feature_2** are created as:

-  :math:`(-\infty, 1.5]`
-  :math:`(1.5, 3.0]`
-  :math:`(3.0, \infty)`

**Note:** The counts provided represent the number of data points (samples) falling into each leaf.

Interval :math:`(-\infty, 1.5]`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Tree 1 Calculation**

- **Leaf Included:**
  - **Leaf 1:** raw_score: 1.25, leaf_count: 50
- **Leaves where feature_2 <= 1.5**, so :math:`D_1` includes Leaf 1.

**Expected Value for Tree 1:**

.. math::

   E[f_1(x) \mid x \in (-\infty, 1.5)] = \frac{1.25 \times 50}{50} = 1.25

**Tree 2 Calculation**

- **Leaves Included:**
  - **Leaf 1:** raw_score: 0.12, leaf_count: 40
  - **Leaf 2:** raw_score: 0.30, leaf_count: 35

Since **Tree 2** splits on **feature_2 <= 3.0**, and our interval is :math:`(-\infty, 1.5]`, both Leaf 1 and Leaf 2 are considered. However, we need to adjust the counts to reflect only the data where **feature_2 <= 1.5**.

Assuming a uniform distribution between :math:`(-\infty, 3.0]`, the proportion of data where **feature_2 <= 1.5** is 50%. Therefore, we adjust the leaf counts:

- **Adjusted Leaf Counts:**
  - **Leaf 1:** 40 × 0.5 = 20
  - **Leaf 2:** 35 × 0.5 = 17.5

**Total Adjusted Count for Tree 2:** 20 + 17.5 = 37.5

**Weighted Sum of Leaf Values:**

.. math::

   \text{Weighted Sum} = (0.12 \times 20) + (0.30 \times 17.5) = 2.4 + 5.25 = 7.65

**Expected Value for Tree 2:**

.. math::

   E[f_2(x) \mid x \in (-\infty, 1.5)] = \frac{7.65}{37.5} = 0.204

**Total Expected Value for Interval** :math:`(-\infty, 1.5]`:

.. math::

   E[F(x) \mid x \in (-\infty, 1.5)] = 1.25 + 0.204 = 1.454

**Average Data Count Across Trees:**

According to the algorithm:

.. math::

   AC[x \in \text{interval}] = \frac{1}{|S_x|} \sum_{t \in S_x} \left( \sum_{i \in D_t} L_{t,i} \right)

- **Set of Trees Using Feature 2:** Both Tree 1 and Tree 2, so :math:`|S_x| = 2`
- **Total Counts in Interval:**
  - **Tree 1:** 50 (Leaf 1)
  - **Tree 2:** 37.5 (Adjusted counts of Leaf 1 and Leaf 2)

.. math::

   AC[x \in (-\infty, 1.5)] = \frac{1}{2} (50 + 37.5) = \frac{87.5}{2} = 43.75

Interval :math:`(1.5, 3.0]`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Tree 1 Calculation**

- **Leaves Included:**
  - **Leaf 2:** raw_score: 1.57, leaf_count: 30
  - **Leaf 3:** raw_score: 2.10, leaf_count: 20

We need to adjust the counts to reflect only the data where **feature_2 ∈ (1.5, 3.0]**. Assuming a uniform distribution between **feature_2 > 1.5**, we can split the counts equally between the intervals **(1.5, 3.0]** and **(3.0, ∞)**.

- **Adjusted Counts:**
  - **Leaf 2:** 30 × 0.5 = 15
  - **Leaf 3:** 20 × 0.5 = 10

**Total Adjusted Count for Tree 1:** 15 + 10 = 25

**Weighted Sum of Leaf Values:**

.. math::

   \text{Weighted Sum} = (1.57 \times 15) + (2.10 \times 10) = 23.55 + 21.0 = 44.55

**Expected Value for Tree 1:**

.. math::

   E[f_1(x) \mid x \in (1.5, 3.0)] = \frac{44.55}{25} = 1.782

**Tree 2 Calculation**

- **Leaves Included:**
  - **Leaf 1:** raw_score: 0.12, leaf_count: 40
  - **Leaf 2:** raw_score: 0.30, leaf_count: 35

Adjusted counts (since **feature_2 ≤ 3.0**):

- **Adjusted Leaf Counts:**
  - **Leaf 1:** 40 × 0.5 = 20
  - **Leaf 2:** 35 × 0.5 = 17.5

**Total Adjusted Count for Tree 2:** 20 + 17.5 = 37.5

**Weighted Sum of Leaf Values:**

.. math::

   \text{Weighted Sum} = (0.12 \times 20) + (0.30 \times 17.5) = 2.4 + 5.25 = 7.65

**Expected Value for Tree 2:**

.. math::

   E[f_2(x) \mid x \in (1.5, 3.0)] = \frac{7.65}{37.5} = 0.204

**Total Expected Value for Interval** :math:`(1.5, 3.0]`:

.. math::

   E[F(x) \mid x \in (1.5, 3.0)] = 1.782 + 0.204 = 1.986

**Average Data Count Across Trees:**

- **Total Counts in Interval:**
  - **Tree 1:** 25 (Adjusted counts)
  - **Tree 2:** 37.5 (Adjusted counts)

.. math::

   AC[x \in (1.5, 3.0)] = \frac{1}{2} (25 + 37.5) = \frac{62.5}{2} = 31.25

Interval :math:`(3.0, \infty)`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Tree 1 Calculation**

- **Leaves Included:**
  - **Leaf 2:** adjusted count = 30 × 0.5 = 15
  - **Leaf 3:** adjusted count = 20 × 0.5 = 10

(Counts are the same as in the previous interval due to equal splitting.)

**Total Adjusted Count for Tree 1:** 15 + 10 = 25

**Weighted Sum of Leaf Values:**

Same as before:

.. math::

   \text{Weighted Sum} = (1.57 \times 15) + (2.10 \times 10) = 44.55

**Expected Value for Tree 1:**

.. math::

   E[f_1(x) \mid x \in (3.0, \infty)] = \frac{44.55}{25} = 1.782

**Tree 2 Calculation**

- **Leaf Included:**
  - **Leaf 3:** raw_score: 0.50, leaf_count: 25

**Expected Value for Tree 2:**

.. math::

   E[f_2(x) \mid x \in (3.0, \infty)] = \frac{0.50 \times 25}{25} = 0.50

**Total Expected Value for Interval** :math:`(3.0, \infty)`:

.. math::

   E[F(x) \mid x \in (3.0, \infty)] = 1.782 + 0.50 = 2.282

**Average Data Count Across Trees:**

- **Total Counts in Interval:**
  - **Tree 1:** 25 (Adjusted counts)
  - **Tree 2:** 25 (Leaf 3)

.. math::

   AC[x \in (3.0, \infty)] = \frac{1}{2} (25 + 25) = 25

Calculation of Overall Expected Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Overall Expected Value**

.. math::

   E[F(x)] = \frac{(1.454 \times 43.75) + (1.9686 \times 31.25) + (2.282 \times 25)}{43.75 + 31.25 + 25} = 1.821

**Overall Expected Value for Tree 2:**

Summary of Results
~~~~~~~~~~~~~~~~~~~

We summarize the expected values and average data counts for each interval:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Interval (:math:`\text{feature}_2` range)
     - Expected Value (:math:`E[F(x) \mid \text{interval}]`)
     - Overall Expected Value (:math:`E[F(x)]`)
     - Difference (:math:`E[F(x) \mid \text{interval}] - E[F(x)]`)
     - Average Data Count (:math:`AC[x \in \text{interval}]`)
   * - :math:`(-\infty, 1.5]`
     - 1.454
     - 1.821
     - -0.367
     - 43.75
   * - :math:`(1.5, 3.0]`
     - 1.986
     - 1.821
     - +0.165
     - 31.25
   * - :math:`(3.0, \infty)`
     - 2.282
     - 1.821
     - +0.461
     - 25

Conclusion
----------

By incorporating leaf counts into our calculations, we follow the treemind algorithm more precisely. The average data counts help us understand the distribution of data across the intervals and ensure that each tree's contribution is weighted appropriately.

- In the interval :math:`(-\infty, 1.5]`, the expected value is lower than the overall expected value, indicating that **feature_2** has a negative contribution in this range.
- In the interval :math:`(1.5, 3.0]`, the expected value is slightly higher than the overall expected value, showing a positive contribution.
- In the interval :math:`(3.0, \infty)`, the expected value is significantly higher, suggesting that higher values of **feature_2** greatly increase the model's prediction.

By calculating both the expected values and average data counts, we gain a comprehensive understanding of how **feature_2** influences the model's predictions across different ranges of data. This detailed analysis allows us to quantify the marginal effect of features accurately, adhering closely to the treemind algorithm's methodology.

The inclusion of leaf counts and average data counts ensures that our calculations reflect the true impact of each feature, weighted by the number of data points in each leaf. This approach minimizes noise and provides a clear picture of feature contributions within specific intervals.

Additional Notes
-----------------

.. note::

   **Mathematical Limitations and Practical Considerations**

   1. **Theoretical Foundation:**
      Although the algorithm produces desired results in practice, it currently lacks formal mathematical proof.

   2. **Interpretation of Results:**
      - The differences obtained by subtracting the final expected value are for interpretative purposes only
      - These values do not reflect the true differences but show correlation with actual values

   3. **Asymptotic Performance:**
      The algorithm's performance improves as both:
      
      - Number of estimators (trees) approaches infinity
      - Amount of data approaches infinity