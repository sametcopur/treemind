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
  - :math:`D_t`: Set of leaves in tree :math:`t` where feature :math:`x` is used as split node in the decision path **and the** :math:`x` **node matches the analysis interval**.

**Matching Intervals:**

When determining :math:`D_t`, we must ensure that the intervals of feature :math:`x` represented by the nodes align with the interval we are analyzing. For instance, if the split points in the tree for feature :math:`x` are at 15, 18, and 20, this creates intervals:

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

This formula computes the weighted average of leaf values, where weights are the leaf counts.

**Average Data Count Across Trees:**

.. math::

   AC[x \in \text{interval}] = \frac{1}{|S_x|} \sum_{t \in S_x} \left( \sum_{i \in D_t} L_{t,i} \right)

**Overall Expected Value:**

The average expected value across all trees using feature :math:`x` is:

.. math::

   E[F(x) \mid x \in \text{interval}] = \frac{\sum_{t \in S_x} E[f_t(x) \mid x \in \text{interval}] \cdot \left( \sum_{i \in D_t} L_{t,i} \right)}{\sum_{t \in S_x} \left( \sum_{i \in D_t} L_{t,i} \right)}

This represents the aggregated contribution of feature :math:`x \in \text{interval}` over all relevant trees.

5. Difference from Mean
~~~~~~~~~~~~~~~~~~~~~~~

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
   
   - Identify split points for `feature_1` to determine its intervals.  
   - Identify split points for `feature_2` to determine its intervals.  

3. **Combined Interval Analysis**

   For each combination of intervals from `feature_1` and `feature_2`, calculate the expected model output:

   .. math::
      E[F(x) \mid x_1 \in \text{interval}_1, x_2 \in \text{interval}_2]

The forward steps remain consistent as described, and this approach can be extended to accommodate additional features.


7. Instance-based Feature Explanations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The algorithm calculates feature contributions for specific instances by analyzing the trees where the feature appears as a split 
node in the instance's prediction path.

1. **Expected Value Calculation Per Tree**

   For a given feature :math:`x` and tree :math:`t`:

   - Let :math:`L_{t,i}` be the leaf count for leaf :math:`i` in tree :math:`t`
   - Let :math:`V_{t,i}` be the prediction value for leaf :math:`i` in tree :math:`t`
   - Let :math:`D_{x,t}` be the set of all leaves in tree :math:`t` where feature :math:`x` is used as a split node (regardless of ranges)

   The expected value for feature :math:`x` in tree :math:`t` is:

   .. math::

      E[f_t(x)] = \frac{\sum_{i \in D_{x,t}} L_{t,i} \cdot V_{t,i}}{\sum_{i \in D_{x,t}} L_{t,i}}

2. **Total Expected Value**

   The total expected value for feature :math:`x` across all trees is simply the sum of individual tree expectations:

   .. math::

      E[F(x)] = \sum_{t=1}^T E[f_t(x)]

3. **Instance-Specific Feature Contribution**

   For a specific instance :math:`i` and feature :math:`x`:
   
   - For each tree :math:`t`, let :math:`P_{t,i}` be the leaf reached during prediction
   - Let :math:`V_{t,i}` be the prediction value of the reached leaf
   - Let :math:`S_{t,x}` be the set of trees where feature :math:`x` is used as a split node in the decision path leading to the leaf.

   The contribution for instance :math:`i` and feature :math:`x` is:

   .. math::

      \text{Contribution}_{i,x} = \sum_{t \in S_{t,x}} V_{t,i}

   If feature :math:`x` is not used as a split node in tree :math:`t`:

   .. math::

      \text{Contribution}_{i,x} = 0

8. Back Data Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

The treemind algorithm allows for the integration of back data, which dynamically updates the leaf counts to reflect the new data while 
keeping the tree structure (splits and leaf values) unchanged.


When new data **back data** is provided, the leaf counts are recalculated as:

.. math::

   L'_{t,i} = \sum_{d \in B} I(d \text{ falls into leaf } i)

where:

- :math:`L'_{t,i}`: Updated leaf count for leaf :math:`i` in tree :math:`t`
- :math:`B`: Set of back data instances
- :math:`I(d \text{ falls into leaf } i)`: Indicator function (1 if instance :math:`d` falls into leaf :math:`i`, 0 otherwise)

This formula completely replaces the original leaf counts with counts derived from the back data.

Example Application
-------------------

To illustrate the treemind algorithm, we will analyze two decision trees with counts for each leaf node to calculate average data counts as specified in the algorithm.

1. Tree Structures
~~~~~~~~~~~~~~~~~~

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

2. Interval Analysis
~~~~~~~~~~~~~~~~~~~~

The split points for feature_2 across both trees are:

- Tree 1: 1.5
- Tree 2: 3.0

This creates the following intervals for analysis:

-  :math:`(-\infty, 1.5]`
-  :math:`(1.5, 3.0]`
-  :math:`(3.0, \infty)`

Note: Both trees use feature_2 as a split feature, so :math:`|S_x| = 2`.

Interval :math:`(-\infty, 1.5]`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Tree 1 Calculation**

- **Matching Leaves:**

  - Leaf 1: raw_score: 1.25, leaf_count: 50
  - This leaf exactly matches our interval

**Expected Value for Tree 1:**

.. math::

   E[f_1(x) \mid x \in (-\infty, 1.5)] = \frac{1.25 \times 50}{50} = 1.25

**Tree 2 Calculation**

- **Matching Leaves:**

  - Both leaves in feature_2 ≤ 3.0 branch are included as (-∞, 1.5] ⊂ (-∞, 3.0]
  - Leaf 1: raw_score: 0.12, leaf_count: 40
  - Leaf 2: raw_score: 0.30, leaf_count: 35

**Expected Value for Tree 2:**

.. math::

   E[f_2(x) \mid x \in (-\infty, 1.5)] = \frac{(0.12 \times 40) + (0.30 \times 35)}{75} = 0.204

**Total Expected Value for Interval** :math:`(-\infty, 1.5]`:

.. math::

   E[F(x) \mid x \in (-\infty, 1.5)] = 1.25 + 0.204 = 1.454

**Average Data Count:**

.. math::

   AC[x \in (-\infty, 1.5)] = \frac{1}{2} (50 + 75) = 62.5

Interval :math:`(1.5, 3.0]`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Tree 1 Calculation**

- **Matching Leaves:**

  - Leaf 2: raw_score: 1.57, leaf_count: 30
  - Leaf 3: raw_score: 2.10, leaf_count: 20
  - These leaves represent feature_2 > 1.5

**Expected Value for Tree 1:**

.. math::

   E[f_1(x) \mid x \in (1.5, 3.0)] = \frac{(1.57 \times 30) + (2.10 \times 20)}{50} = 1.782

**Tree 2 Calculation**

- **Matching Leaves:**

  - Same leaves as (-∞, 1.5] interval since (1.5, 3.0] ⊂ (-∞, 3.0]
  - Leaf 1: raw_score: 0.12, leaf_count: 40
  - Leaf 2: raw_score: 0.30, leaf_count: 35

**Expected Value for Tree 2:**

.. math::

   E[f_2(x) \mid x \in (1.5, 3.0)] = \frac{(0.12 \times 40) + (0.30 \times 35)}{75} = 0.204

**Total Expected Value for Interval** :math:`(1.5, 3.0]`:

.. math::

   E[F(x) \mid x \in (1.5, 3.0)] = 1.782 + 0.204 = 1.986

**Average Data Count:**

.. math::

   AC[x \in (1.5, 3.0)] = \frac{1}{2} (50 + 75) = 62.5

Interval :math:`(3.0, \infty)`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Tree 1 Calculation**

- **Matching Leaves:**

  - Same leaves as (1.5, 3.0] since they represent feature_2 > 1.5
  - Leaf 2: raw_score: 1.57, leaf_count: 30
  - Leaf 3: raw_score: 2.10, leaf_count: 20

**Expected Value for Tree 1:**

.. math::

   E[f_1(x) \mid x \in (3.0, \infty)] = \frac{(1.57 \times 30) + (2.10 \times 20)}{50} = 1.782

**Tree 2 Calculation**

- **Matching Leaves:**

  - Leaf 3: raw_score: 0.50, leaf_count: 25
  - This leaf exactly matches our interval

**Expected Value for Tree 2:**

.. math::

   E[f_2(x) \mid x \in (3.0, \infty)] = \frac{0.50 \times 25}{25} = 0.50

**Total Expected Value for Interval** :math:`(3.0, \infty)`:

.. math::

   E[F(x) \mid x \in (3.0, \infty)] = 1.782 + 0.50 = 2.282

**Average Data Count:**

.. math::

   AC[x \in (3.0, \infty)] = \frac{1}{2} (50 + 25) = 37.5

3. Overall Expected Value Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the formula:

.. math::

   E[F(x)] = \frac{\sum_{\text{interval}} E[F(x) \mid x \in \text{interval}] \cdot AC[x \in \text{interval}]}{\sum_{\text{interval}} AC[x \in \text{interval}]}

We get:

.. math::

   E[F(x)] = \frac{(1.454 \times 62.5) + (1.986 \times 62.5) + (2.282 \times 37.5)}{62.5 + 62.5 + 37.5} = 1.821

4. Summary of Results
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Interval
     - Expected Value
     - Overall Expected Value
     - Difference
     - Average Data Count
   * - (-∞, 1.5]
     - 1.454
     - 1.821
     - -0.367
     - 62.5
   * - (1.5, 3.0]
     - 1.986
     - 1.821
     - +0.165
     - 62.5
   * - (3.0, ∞)
     - 2.282
     - 1.821
     - +0.461
     - 37.5

- **Interval (-∞, 1.5]:**

  - **Difference:** -0.367 (negative contribution; below the overall model expectation).
  - **Interpretation:** Feature_2 in this range reduces the model's output compared to the average.

- **Interval (1.5, 3.0]:**

  - **Difference:** +0.165 (moderate positive contribution).
  - **Interpretation:** Feature_2 in this range slightly increases the model's output compared to the average.

- **Interval (3.0, ∞):**

  - **Difference:** +0.461 (strong positive contribution).
  - **Interpretation:** Feature_2 in this range significantly boosts the model's output compared to the average.


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