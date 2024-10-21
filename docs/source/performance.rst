Evaluating Treemind's Performance
===================================================

This document outlines the process of generating synthetic data and evaluating the Treemind model's performance in detecting individual feature effects and interactions. The results show strong alignment between the actual data and Treemind's predictions, indicating the model's ability to capture complex relationships.

The visualizations presented in this document are based on the results from the Jupyter Notebook available at the following link: 

`Treemind Results on GitHub <https://github.com/treemind/results.ipynb>`_


Comparison of Actual vs. Treemind Predictions
------------------------------------------------------

To evaluate Treemind's performance, we compare the actual feature relationships with the model's predictions for various features and interactions. The following sections present line plots and scatter plots showing the actual values versus Treemind's predicted values.

1. **Feature 4: Actual vs. Treemind Analysis**

   The line plots below compare the actual values of ``feature_4`` with the Treemind model's predictions. The high degree of similarity between the two lines suggests that Treemind accurately captures the behavior of ``feature_4``.

   .. image:: _static/performance/feature_real.png
      :alt: Actual values for feature 4
      :width: 600px

   .. image:: _static/performance/feature_pred.png
      :alt: Treemind predicted values for feature 4
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
