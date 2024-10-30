treemind Algorithm
========================

**Tree 1:**

.. code-block:: none

   |--- feature_2 <= 1.5
   |   |--- raw_score: 1.25
   |--- feature_2 >  1.5
   |   |--- feature_1 <= 2.5
   |   |   |--- raw_score: 1.57
   |   |--- feature_1 >  2.5
   |   |   |--- raw_score: 2.10

**Tree 2:**

.. code-block:: none

   |--- feature_2 <= 3.0
   |   |--- feature_1 <= 1.0
   |   |   |--- raw_score: 0.12
   |   |--- feature_1 >  1.0
   |   |   |--- raw_score: 0.30
   |--- feature_2 >  3.0
   |   |--- raw_score: 0.50



Algorithm 1
^^^^^^^^^^^

The first algorithm explains what kind of results the model would produce based on split points. It analyzes how different feature split points 
affect the outcome, giving insights into the general decision boundaries and patterns created by the model.

The split points for `feature_2` across both trees are:

- `(-inf, 1.5, 3.0, inf)`

**Key Terms**

- **Raw Score**: This is the output value at each leaf in the tree, representing the score or prediction assigned when a data point falls into that specific leaf.

- **Leaf Count**: This represents the number of data points (samples) that fall into each leaf during training. `leaf_count` effectively  indicates how many data points "vote" for the prediction in that leaf. By weighting raw scores based on `leaf_count`, we make predictions more representative of the underlying data distribution.


**Example Interval:** ``(-inf, 1.5]``

For this example, we calculate predictions in the interval where ``feature_2`` is less than or equal to 1.5.

**Tree 1 Calculation**

In Tree 1, the rule for this interval is as follows:

   - When ``feature_2 <= 1.5``, the prediction is:
     - ``raw_score`` = 1.25
     - ``leaf_count`` = 20

Since there is only one leaf in **Tree 1** for this interval, the prediction result for  **Tree 1** here remains:

.. math::

   \text{Tree 1 Result} = 1.25

**Tree 2 Calculation**

In **Tree 2**, for ``feature_2 <= 1.5``, we have two possible outcomes depending on ``feature_1``:

   - **Leaf 1**: if ``feature_1 <= 1.0``
     - ``raw_score`` = 0.12
     - ``leaf_count`` = 15
   - **Leaf 2**: if ``feature_1 > 1.0``
     - ``raw_score`` = 0.30
     - ``leaf_count`` = 20

The weighted result for **Tree 2**, considering both leaves, is calculated as follows:

.. math::

   \text{Tree 2 Result} = \frac{(0.12 \times 15) + (0.30 \times 20)}{15 + 20} = \frac{1.8 + 6}{35} = 0.22

**Final Prediction for Interval** ``(-inf, 1.5)``

Since we are using a boosting approach, we sum the results from Tree 1 and Tree 2 to get the final prediction for this interval:

.. math::

   \text{Total Prediction} = 1.25 + 0.22 = 1.47

Repeating this process across all intervals, ``(-inf, 1.5)``, ``(1.5, 3.0)``, and ``(3.0, inf)``, yields predictions for each range 
that are accurately weighted and boosted. This method enhances prediction quality by respecting both the depth of each tree's 
predictions and the cumulative adjustments from boosting.

This process is also applied to **identify interactions between features**. For instance, in this example, we analyze split points 
for both ``feature_1`` and ``feature_2`` by examining each in turn. When ``feature_2`` is in the range ``(-inf, 1.5)``, we further 
analyze the effect of ``feature_1`` within its own range, such as ``(-inf, 1.0)``. By repeating this approach for feature combinations, 
we uncover potential interactions that contribute to a more nuanced and accurate prediction.


**Expected Value Calculation**:

After analyzing the intervals and interactions, we calculate expected values to understand feature contributions to predictions. 
The calculation method is similar for both single feature and feature interaction analysis, with only a notation difference to 
distinguish between them.


For single feature analysis, we use :math:`E[x]`, while for feature interactions, we use :math:`E[x,y]`. Both are calculated using 
the raw scores and counts from each leaf:

.. math::

   E[x] \text{ or } E[x,y] = \frac{\sum (\text{val} \times \text{count})}{\sum \text{count}}

The adjusted score is then calculated by subtracting the appropriate expected value:

.. math::

   \text{adjusted_val} = \text{val} - E[x] \text{ or } \text{val} - E[x,y]

This calculation helps capture both individual feature effects and their interactions, providing insight into how features contribute 
to the model's predictions.

Algorithm 2
^^^^^^^^^^^

The second algorithm explains the model’s decision-making process for a specific input. By examining which features influenced the final 
leaf for a given sample, it sheds light on what the model relied on for that particular decision, providing a more detailed view of 
feature importance for individual predictions.

Suppose we have a sample with the following feature values:

   - ``feature_2 = 1.0``
   - ``feature_1 = 1.5``

Using these values, we can determine which leaf each tree directs the sample to.

1. **Tree 1 Path**:
   - ``feature_2 = 1.0`` is less than or equal to 1.5, so the sample reaches the leaf with ``raw_score = 1.25``.

   Therefore, in Tree 1, the sample falls into the leaf with a ``raw_score`` of 1.25.

2. **Tree 2 Path**:
   - ``feature_2 = 1.0`` is less than or equal to 3.0, so the sample moves down the left branch.
   - ``feature_1 = 1.5`` is greater than 1.0, so the sample reaches the leaf with ``raw_score = 0.30``.

   Thus, in Tree 2, the sample falls into the leaf with a ``raw_score`` of 0.30.

**Determining Feature Impact**

Next, we evaluate if ``feature_1`` is part of the decision path in each tree.

- **Tree 1**: The leaf with ``raw_score = 1.25`` is reached without considering ``feature_1`` in the decision path (only ``feature_2`` is used).
- **Tree 2**: The leaf with ``raw_score = 0.30`` is influenced by ``feature_1`` (``feature_1 > 1.0``).

Since only the leaf in Tree 2 involves ``feature_1`` in its decision path, the total impact of ``feature_1`` for this sample is the raw 
score from Tree 2’s relevant leaf:

.. math::

   \text{Total Score (for feature_1)} = 0.30

By following this process for each sample in the dataset, the algorithm isolates the contribution of ``feature_1`` by summing the raw scores 
of the relevant leaves. This method allows us to quantify how much ``feature_1`` impacts the model’s predictions based on actual data paths 
through the trees.

When the number of estimators in the ensemble is large and the trees are deep, this method becomes significantly more meaningful. With a 
higher estimator count and more complex trees, the model captures finer interactions and dependencies, leading to more accurate and nuanced 
interpretations of feature impact.

If multiple data points are provided, the algorithm calculates the average contribution for each feature across all samples, offering a broader 
view of each feature's importance in the model.