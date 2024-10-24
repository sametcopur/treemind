Evaluating Treemind's Performance
===================================================

This document outlines the process of generating synthetic data and evaluating the Treemind model's performance in detecting individual feature effects and interactions. The results show strong alignment between the actual data and Treemind's predictions, indicating the model's ability to capture complex relationships.

The visualizations presented in this document are based on the results from the Jupyter Notebook available at the following link: 

`Treemind Results on GitHub <https://github.com/treemind/results.ipynb>`_

Data Transformations and Target Variable Definition
---------------------------------------------------

feature_4 follows a uniform distribution between -2 and 2, while the remaining features are normally distributed.

The transformations and interactions for the features are defined as follows:

1. **Feature Transformations:**

   .. math::

      \text{transformed_0} = (\text{feature_0} - 2)^2

   .. math::

      \text{transformed_1} = \text{feature_1} \cdot \sin(\text{feature_1}) + 1

   .. math::

      \text{transformed_2} = \log(|\text{feature_2}| + 1) \cdot \cos(\text{feature_2})

   .. math::

      \text{transformed_3} = \frac{e^{\text{feature_3}}}{1 + e^{-\text{feature_3}}}

   .. math::

      \text{transformed_4} = \sqrt{|\text{feature_4}|}

2. **Interactions Between Features:**

   - Interaction between :math:`\text{transformed_0}` and :math:`\text{transformed_1}`:

   .. math::

      \text{interaction_0_1} = 
      \begin{cases}
      \text{transformed_1}, & \text{if } \text{transformed_0} > 1 \\
      -\text{transformed_1}, & \text{otherwise}
      \end{cases}

   - Interaction between :math:`\text{transformed_2}` and :math:`\text{transformed_3}`:

   .. math::

      \text{interaction_2_3} = \text{transformed_2} \cdot \text{transformed_3}

3. **Target Variable Construction:**

   The target variable combines the interactions and transformations to create a complex relationship. Coefficients are used to adjust the influence of each interaction, and Gaussian noise is added to simulate measurement error.

   .. math::

      \text{target} = \text{transformed_0} + \text{interaction_0_1} 
      + \text{interaction_2_3} + \text{transformed_4} + \mathcal{N}(0, 0.1)

""""

Comparison of Actual vs. Treemind Predictions
--------------------------------------------------

To evaluate Treemind's performance, we compare the actual feature relationships with the model's predictions for various features and interactions. The following sections present line plots and scatter plots showing the actual values versus Treemind's predicted values.

1. **Feature 0: Actual vs. Treemind Analysis**

   The value of ``feature_0`` influences the target variable through both ``transformed_0`` and ``interaction_0_1``. If we create a line plot comparing ``feature_0`` with the sum of ``transformed_0`` and ``interaction_0_1``, we get the following result:

   .. image:: _static/performance/feature_0_real.png
      :alt: Contribution of feature 0 (transformed_0 + interaction_0_1)
      :width: 600px

   By using the Treemind model to extract the effect of ``feature_0`` from the target, we obtain this visualization:

   .. image:: _static/performance/feature_0_pred.png
      :alt: Treemind predicted values with feature 0 effect extracted
      :width: 600px

1. **Feature 4: Actual vs. Treemind Analysis**

   The value of ``feature_4`` influences the target variable solely through ``transformed_4``. If we create a line plot showing the relationship between ``feature_4`` and the transformed component, we get the following visualization:

   .. image:: _static/performance/feature_real.png
      :alt: Contribution of feature 4 (transformed_4)
      :width: 600px

   By using the Treemind model to extract the effect of ``feature_4`` from the target, we obtain this visualization:

   .. image:: _static/performance/feature_pred.png
      :alt: Treemind predicted values with feature 4 effect extracted
      :width: 600px

2. **Interaction between Feature 1 and Feature 2: Actual vs. Treemind Analysis**

   The scatter plots below compare the actual interaction values between ``feature_1`` and ``feature_2`` with the Treemind model's predictions. The close alignment between the actual and predicted values indicates Treemind's effectiveness in modeling this interaction.

   .. image:: _static/performance/test_1_real.png
      :alt: Actual interaction values between feature 1 and feature 2
      :width: 600px

   .. image:: _static/performance/test_1_pred.png
      :alt: Treemind predicted interaction values between feature 1 and feature 2
      :width: 600px

3. **Interaction between Feature 2 and Feature 3: Actual vs. Treemind Analysis**

   The following scatter plots compare the actual interaction values between ``feature_2`` and ``feature_3`` with Treemind's predictions. The plots show a high level of agreement, further validating Treemind's ability to model feature interactions accurately.

   .. image:: _static/performance/test_2_real.png
      :alt: Actual interaction values between feature 2 and feature 3
      :width: 600px

   .. image:: _static/performance/test_2_pred.png
      :alt: Treemind predicted interaction values between feature 2 and feature 3
      :width: 600px

Conclusion
----------

Treemind evaluates features by considering their global impact rather than isolating them in their individual or interaction effects. When analyzing the interaction between ``feature_1`` and ``feature_2``, Treemind takes into account the broader context, including other interactions involving these features. It assesses how these two features interact while considering their relationships with the rest of the features. Similarly, when evaluating a single feature, Treemind does not isolate its effect but instead looks at how it influences the model's predictions within the context of the overall feature set. This approach allows for a more comprehensive understanding of feature importance and interactions.

The close alignment observed in the plots suggests that Treemind is effective at modeling both individual feature effects and complex feature interactions.
