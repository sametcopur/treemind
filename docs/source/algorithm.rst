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


**Step 1: Find Split Points for `feature_2`** 

The split points for `feature_2` across both trees are:

- `(-inf, 1.5, 3.0, inf)`

**Step 2: Analyze the Interval `(-inf, 1.5)`**  


1. **Identify Possible Leaves in Each Tree:**

   - For **Tree 1**, when ``feature_2`` is in the range ``(-inf, 1.5)``, the possible leaf has a ``raw_score`` of ``1.25``.
   - For **Tree 2**, when ``feature_2`` is in the range ``(-inf, 1.5)``, the possible leaves can have:

     - ``raw_score`` of ``0.12`` if ``feature_1 <= 1.0``
     - ``raw_score`` of ``0.30`` if ``feature_1 > 1.0``


2. **Calculate Combinations of Raw Scores:**

   - **Combination 1:** Tree 1 (``1.25``) + Tree 2 (``0.12``) = ``1.25 + 0.12 = 1.37``
   - **Combination 2:** Tree 1 (``1.25``) + Tree 2 (``0.30``) = ``1.25 + 0.30 = 1.55``


3. **Average the Results for the Interval** ``(-inf, 1.5)``

   - The average outcome for this interval is:

     .. math::

        \text{Average} = \frac{1.37 + 1.55}{2} = 1.46

Therefore, for ``feature_2`` in the range ``(-inf, 1.5)``, the predicted average ``raw_score`` is ``1.46``.

This process can be repeated for the other intervals ``(1.5, 3.0)`` and ``(3.0, inf)`` to calculate the predicted outcomes for those ranges based on the raw scores from both trees.