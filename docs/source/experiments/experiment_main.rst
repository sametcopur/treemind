Evaluating treemind's Performance
=================================

treemind is an experimental approach to feature and feature interaction analysis, 
developed based on practical observations rather than theoretical foundations. 
Our extensive testing reveals intriguing patterns in its behavior that warrant detailed examination.

The algorithm's performance exhibits variability—even with similarly structured synthetic data, 
treemind can produce both meaningful and less interpretable results. This variability arises primarily 
from its reliance on the model's underlying tree structure. In some cases, it successfully isolates 
the cumulative effects of features, while in others, it produces results that are more challenging to interpret.

treemind focuses on analyzing how a feature behaves under specific intervals and explains its impact 
on model predictions. In single-feature analysis, treemind generally aligns closely with results obtained 
from widely used libraries like SHAP.

When analyzing pairwise feature interactions, treemind examines the cumulative effects of both features 
on the model's predictions under specific conditions. For instance, if the model's behavior is influenced 
by a target function like ``(x1 - x2)^2``, treemind captures not only ``-2x1x2`` but also the total effect, 
including ``x1^2`` and ``x2^2``.


treemind is exceptionally fast in terms of computational time, particularly excelling in interaction analysis. 
It can analyze the entire model within seconds.

Experiments
-----------

Data Generation
^^^^^^^^^^^^^^^^

The datasets used in this analysis were synthetically generated to adhere to specific distributions. Features were labeled as ``feature_0, feature_1, ..., feature_n`` for clarity and ease of reference. 

To introduce complexity, transformations—denoted as ``transform_0, transform_1, ..., transform_n``—were applied to these features. Each transformation was expressed as a function of its corresponding feature. For example:

.. math::

   \text{transform_i} = f(\text{feature_i})


Next, feature interactions, such as ``interaction_0_1``, were created by combining these transformations. For example:

.. math::

   \text{interaction_i_j} = g(\text{transform_i}, \text{transform_j})

The target variable was then generated using a combination of these components. You can check how the target was created at the beginning of each experiment.

Value Adjustment
^^^^^^^^^^^^^^^^^

The adjusted values were computed to ensure that comparisons between truth values and (treemind and shap) analyses are both fair and meaningful. 
By centering the predictions and target function around their respective means, we isolate the relative contributions.

For the truth values, the adjustment was applied as follows:

.. math::

   \text{truth_adjusted} = \text{truth} - \text{mean}(\text{truth})

The adjusted predicted values were included to ensure that treemind does not achieve over-success by capturing relationships that are already visually evident in the raw predictions. Similarly, for the predicted values:

.. math::

   \text{predicted_adjusted} = \text{predicted} - \text{mean}(\text{predicted})


Single-Feature Analysis
^^^^^^^^^^^^^^^^^^^^^^^

For single-feature analyses, we visualized the results using line and scatter plots:

- The **x-axis** represented feature values.
- The **y-axis** varied depending on the context:

  - **SHAP values** derived from SHAP-based methods.
  - **treemind interaction values** as computed by the treemind algorithm.
  - **Adjusted prediction values** indicating the raw model output.
  - **Adjusted truth values**, which are the ground truth for the target variable.

Pairwise Interaction Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For pairwise interactions, scatter plots were utilized to visualize the relationship between two interacting features and their cumulative impact on the model's predictions:

- The **x-axis** and **y-axis** represent the values of two interacting features.
- A **color bar** indicates the effect on predictions:

  - **SHAP interaction values** for SHAP-based analyses.
  - **treemind interaction values** as computed by treemind.
  - **Adjusted prediction values** indicating the raw model output.
  - **Adjusted truth values**, which are the ground truth for the target variable.

For details on the experiment design and target functions used, refer to the experiment setup documentation:

`treemind Experiments <https://github.com/sametcopur/treemind/blob/main/examples/>`_
 