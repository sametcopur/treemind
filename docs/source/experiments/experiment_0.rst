Experiment 2
============

.. code-block:: python

   target = (
      0.4 * transformed_0
      - 0.6 * transformed_1
      + 0.3 * transformed_2
      + 0.5 * transformed_3
      - 0.4 * transformed_4
      + 0.7 * transformed_5
      - 0.3 * transformed_6
      + 0.5 * transformed_7
      - 0.4 * transformed_8
      + 0.6 * interaction_0_1
      - 0.5 * interaction_2_3
      + 0.4 * interaction_4_5
      - 0.3 * interaction_6_7
      + np.random.normal(loc=0, scale=0.2, size=n_samples)
    )

Feature Analysis
^^^^^^^^^^^^^^^^^

feature_0
""""""""""

**Function plot:**

.. image:: ../_static/experiments/experiment_2/feature_0_real.png
   :alt: Contribution of feature 0
   :width: 600px

**treemind plot:**

.. image:: ../_static/experiments/experiment_2/feature_0_treemind.png
   :alt: treemind's extracted values for feature 0
   :width: 600px

**SHAP plot:**

.. image:: ../_static/experiments/experiment_2/feature_0_shap.png
   :alt: SHAP values for feature 0
   :width: 600px

feature_1
""""""""""

**Function plot:**

.. image:: ../_static/experiments/experiment_2/feature_1_real.png
   :alt: Contribution of feature 1
   :width: 600px

**treemind plot:**

.. image:: ../_static/experiments/experiment_2/feature_1_treemind.png
   :alt: treemind's extracted values for feature 1
   :width: 600px

**SHAP plot:**

.. image:: ../_static/experiments/experiment_2/feature_1_shap.png
   :alt: SHAP values for feature 1
   :width: 600px

feature_2
""""""""""

**Function plot:**

.. image:: ../_static/experiments/experiment_2/feature_2_real.png
   :alt: Contribution of feature 2
   :width: 600px

**treemind plot:**

.. image:: ../_static/experiments/experiment_2/feature_2_treemind.png
   :alt: treemind's extracted values for feature 2
   :width: 600px

**SHAP plot:**

.. image:: ../_static/experiments/experiment_2/feature_2_shap.png
   :alt: SHAP values for feature 2
   :width: 600px

feature_3
""""""""""

**Function plot:**

.. image:: ../_static/experiments/experiment_2/feature_3_real.png
   :alt: Contribution of feature 3
   :width: 600px

**treemind plot:**

.. image:: ../_static/experiments/experiment_2/feature_3_treemind.png
   :alt: treemind's extracted values for feature 3
   :width: 600px

**SHAP plot:**

.. image:: ../_static/experiments/experiment_2/feature_3_shap.png
   :alt: SHAP values for feature 3
   :width: 600px

feature_4
""""""""""

**Function plot:**

.. image:: ../_static/experiments/experiment_2/feature_4_real.png
   :alt: Contribution of feature 4
   :width: 600px

**treemind plot:**

.. image:: ../_static/experiments/experiment_2/feature_4_treemind.png
   :alt: treemind's extracted values for feature 4
   :width: 600px

**SHAP plot:**

.. image:: ../_static/experiments/experiment_2/feature_4_shap.png
   :alt: SHAP values for feature 4
   :width: 600px

feature_5  
""""""""""

**Function plot:** 

.. image:: ../_static/experiments/experiment_2/feature_5_real.png  
   :alt: Contribution of feature 5  
   :width: 600px  

**treemind plot:**  

.. image:: ../_static/experiments/experiment_2/feature_5_treemind.png  
   :alt: treemind's extracted values for feature 5  
   :width: 600px  

**SHAP plot:**  

.. image:: ../_static/experiments/experiment_2/feature_5_shap.png  
   :alt: SHAP values for feature 5  
   :width: 600px  


feature_6  
""""""""""

**Function plot:**  

.. image:: ../_static/experiments/experiment_2/feature_6_real.png  
   :alt: Contribution of feature 6  
   :width: 600px  

**treemind plot:**  

.. image:: ../_static/experiments/experiment_2/feature_6_treemind.png  
   :alt: treemind's extracted values for feature 6  
   :width: 600px  

**SHAP plot:**  

.. image:: ../_static/experiments/experiment_2/feature_6_shap.png  
   :alt: SHAP values for feature 6  
   :width: 600px  


feature_7  
""""""""""

**Function plot:**  

.. image:: ../_static/experiments/experiment_2/feature_7_real.png  
   :alt: Contribution of feature 7  
   :width: 600px  

**treemind plot:**  

.. image:: ../_static/experiments/experiment_2/feature_7_treemind.png  
   :alt: treemind values for feature 7  
   :width: 600px 

**SHAP plot:**  

.. image:: ../_static/experiments/experiment_2/feature_7_shap.png  
   :alt: SHAP values for feature 7  
   :width: 600px  


Interaction Analysis
^^^^^^^^^^^^^^^^^^^^^

feature_0 - feature_1 
"""""""""""""""""""""

**Prediction plot:**  

.. image:: ../_static/experiments/experiment_2/interaction_0_1_pred.png  
   :alt: Prediction values between feature 0 and feature 1
   :width: 600px  

**Function plot:**  

.. image:: ../_static/experiments/experiment_2/interaction_0_1_real_1.png  
   :alt: Actual interaction values between feature 0 and feature 1
   :width: 600px  

.. image:: ../_static/experiments/experiment_2/interaction_0_1_real_2.png  
   :alt: Actual interaction values between feature 0 and feature 1
   :width: 600px  

**treemind plot:**  

.. image:: ../_static/experiments/experiment_2/interaction_0_1_treemind.png  
   :alt: treemind interaction values between feature 0 and feature 1
   :width: 600px  

**SHAP plot:**  

.. image:: ../_static/experiments/experiment_2/interaction_0_1_shap.png  
   :alt: SHAP interaction values between feature 0 and feature 1
   :width: 600px  

feature_2 - feature_3  
"""""""""""""""""""""""

**Prediction plot:**  

.. image:: ../_static/experiments/experiment_2/interaction_2_3_pred.png  
   :alt: Prediction values between feature 2 and feature 3
   :width: 600px  

**Function plot:**  

.. image:: ../_static/experiments/experiment_2/interaction_2_3_real_1.png  
   :alt: Actual interaction values between feature 2 and feature 3  
   :width: 600px  


.. image:: ../_static/experiments/experiment_2/interaction_2_3_real_2.png  
   :alt: Actual interaction values between feature 2 and feature 3  
   :width: 600px  


**treemind plot:**  

.. image:: ../_static/experiments/experiment_2/interaction_2_3_treemind.png  
   :alt: treemind interaction values between feature 2 and feature 3  
   :width: 600px  

**SHAP plot:**  

.. image:: ../_static/experiments/experiment_2/interaction_2_3_shap.png  
   :alt: SHAP interaction values between feature 2 and feature 3  
   :width: 600px  

feature_4 - feature_5  
"""""""""""""""""""""""

**Prediction plot:**  

.. image:: ../_static/experiments/experiment_2/interaction_4_5_pred.png  
   :alt: Prediction values between feature 4 and feature 5
   :width: 600px  

**Function plot:**  

.. image:: ../_static/experiments/experiment_2/interaction_4_5_real_1.png  
   :alt: Actual interaction values between feature 4 and feature 5  
   :width: 600px  

.. image:: ../_static/experiments/experiment_2/interaction_4_5_real_2.png  
   :alt: Actual interaction values between feature 4 and feature 5  
   :width: 600px  


**treemind plot:**  

.. image:: ../_static/experiments/experiment_2/interaction_4_5_treemind.png  
   :alt: treemind interaction values between feature 4 and feature 5  
   :width: 600px  

**SHAP plot:**  

.. image:: ../_static/experiments/experiment_2/interaction_4_5_shap.png  
   :alt: SHAP interaction values between feature 4 and feature 5  
   :width: 600px  

feature_6 - feature_7  
"""""""""""""""""""""""

**Prediction plot:**  

.. image:: ../_static/experiments/experiment_2/interaction_6_7_pred.png  
   :alt: Prediction values between feature 6 and feature 7
   :width: 600px  

**Function plot:**  

.. image:: ../_static/experiments/experiment_2/interaction_6_7_real_1.png  
   :alt: Actual interaction values between feature 6 and feature 7  
   :width: 600px  

.. image:: ../_static/experiments/experiment_2/interaction_6_7_real_2.png  
   :alt: Actual interaction values between feature 6 and feature 7
   :width: 600px  

**treemind plot:**  

.. image:: ../_static/experiments/experiment_2/interaction_6_7_treemind.png  
   :alt: treemind interaction values between feature 6 and feature 7  
   :width: 600px  

**SHAP plot:**  

.. image:: ../_static/experiments/experiment_2/interaction_6_7_shap.png  
   :alt: SHAP interaction values between feature 6 and feature 7  
   :width: 600px  
