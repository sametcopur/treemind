Experiment 1
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
      + 0.6 * interaction_0_2
      - 0.5 * interaction_1_3
      + 0.4 * interaction_4_6
      - 0.3 * interaction_5_7
      + 0.5 * interaction_6_8
      + np.random.normal(loc=0, scale=0.2, size=n_samples)
   )

Feature Analysis
^^^^^^^^^^^^^^^^^

feature_0
""""""""""

**Function plot:**

.. image:: ../_static/experiments/experiment_1/feature_0_real.png
   :alt: Contribution of feature 0
   :width: 600px

**treemind plot:**

.. image:: ../_static/experiments/experiment_1/feature_0_treemind.png
   :alt: treemind's extracted values for feature 0
   :width: 600px

**SHAP plot:**

.. image:: ../_static/experiments/experiment_1/feature_0_shap.png
   :alt: SHAP values for feature 0
   :width: 600px

feature_1
""""""""""

**Function plot:**

.. image:: ../_static/experiments/experiment_1/feature_1_real.png
   :alt: Contribution of feature 1
   :width: 600px

**treemind plot:**

.. image:: ../_static/experiments/experiment_1/feature_1_treemind.png
   :alt: treemind's extracted values for feature 1
   :width: 600px

**SHAP plot:**

.. image:: ../_static/experiments/experiment_1/feature_1_shap.png
   :alt: SHAP values for feature 1
   :width: 600px

feature_2
""""""""""

**Function plot:**

.. image:: ../_static/experiments/experiment_1/feature_2_real.png
   :alt: Contribution of feature 2
   :width: 600px

**treemind plot:**

.. image:: ../_static/experiments/experiment_1/feature_2_treemind.png
   :alt: treemind's extracted values for feature 2
   :width: 600px

**SHAP plot:**

.. image:: ../_static/experiments/experiment_1/feature_2_shap.png
   :alt: SHAP values for feature 2
   :width: 600px

feature_3
""""""""""

**Function plot:**

.. image:: ../_static/experiments/experiment_1/feature_3_real.png
   :alt: Contribution of feature 3
   :width: 600px

**treemind plot:**

.. image:: ../_static/experiments/experiment_1/feature_3_treemind.png
   :alt: treemind's extracted values for feature 3
   :width: 600px

**SHAP plot:**

.. image:: ../_static/experiments/experiment_1/feature_3_shap.png
   :alt: SHAP values for feature 3
   :width: 600px

feature_4
""""""""""

**Function plot:**

.. image:: ../_static/experiments/experiment_1/feature_4_real.png
   :alt: Contribution of feature 4
   :width: 600px

**treemind plot:**

.. image:: ../_static/experiments/experiment_1/feature_4_treemind.png
   :alt: treemind's extracted values for feature 4
   :width: 600px

**SHAP plot:**

.. image:: ../_static/experiments/experiment_1/feature_4_shap.png
   :alt: SHAP values for feature 4
   :width: 600px

feature_5  
""""""""""

**Function plot:** 

.. image:: ../_static/experiments/experiment_1/feature_5_real.png  
   :alt: Contribution of feature 5  
   :width: 600px  

**treemind plot:**  

.. image:: ../_static/experiments/experiment_1/feature_5_treemind.png  
   :alt: treemind's extracted values for feature 5  
   :width: 600px  

**SHAP plot:**  

.. image:: ../_static/experiments/experiment_1/feature_5_shap.png  
   :alt: SHAP values for feature 5  
   :width: 600px  


feature_6  
""""""""""

**Function plot:**  

.. image:: ../_static/experiments/experiment_1/feature_6_real.png  
   :alt: Contribution of feature 6  
   :width: 600px  

**treemind plot:**  

.. image:: ../_static/experiments/experiment_1/feature_6_treemind.png  
   :alt: treemind's extracted values for feature 6  
   :width: 600px  

**SHAP plot:**  

.. image:: ../_static/experiments/experiment_1/feature_6_shap.png  
   :alt: SHAP values for feature 6  
   :width: 600px  


feature_7  
""""""""""

**Function plot:**  

.. image:: ../_static/experiments/experiment_1/feature_7_real.png  
   :alt: Contribution of feature 7  
   :width: 600px  

**treemind plot:**  

.. image:: ../_static/experiments/experiment_1/feature_7_treemind.png  
   :alt: treemind values for feature 7  
   :width: 600px 

**SHAP plot:**  

.. image:: ../_static/experiments/experiment_1/feature_7_shap.png  
   :alt: SHAP values for feature 7  
   :width: 600px  

feature_8
""""""""""

**Function plot:**

.. image:: ../_static/experiments/experiment_1/feature_8_real.png
   :alt: Contribution of feature 8
   :width: 600px

**treemind plot:**

.. image:: ../_static/experiments/experiment_1/feature_8_treemind.png
   :alt: treemind's extracted values for feature 8
   :width: 600px

**SHAP plot:**

.. image:: ../_static/experiments/experiment_1/feature_8_shap.png
   :alt: SHAP values for feature 8
   :width: 600px



Interaction Analysis
^^^^^^^^^^^^^^^^^^^^^

feature_0 - feature_2 
"""""""""""""""""""""

**Prediction plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_0_2_pred.png  
   :alt: Prediction values between feature 0 and feature 2
   :width: 600px  

**Function plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_0_2_real_1.png  
   :alt: Actual interaction values between feature 0 and feature 2
   :width: 600px  

.. image:: ../_static/experiments/experiment_1/interaction_0_2_real_2.png  
   :alt: Actual interaction values between feature 0 and feature 2
   :width: 600px  

**treemind plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_0_2_treemind.png  
   :alt: treemind interaction values between feature 0 and feature 2
   :width: 600px  

**SHAP plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_0_2_shap.png  
   :alt: SHAP interaction values between feature 0 and feature 2
   :width: 600px  

feature_1 - feature_3  
"""""""""""""""""""""""

**Prediction plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_1_3_pred.png  
   :alt: Prediction values between feature 1 and feature 3
   :width: 600px  

**Function plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_1_3_real_1.png  
   :alt: Actual interaction values between feature 1 and feature 3  
   :width: 600px  


.. image:: ../_static/experiments/experiment_1/interaction_1_3_real_2.png  
   :alt: Actual interaction values between feature 1 and feature 3  
   :width: 600px  


**treemind plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_1_3_treemind.png  
   :alt: treemind interaction values between feature 1 and feature 3  
   :width: 600px  

**SHAP plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_1_3_shap.png  
   :alt: SHAP interaction values between feature 1 and feature 3  
   :width: 600px  

feature_4 - feature_6 
"""""""""""""""""""""""

**Prediction plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_4_6_pred.png  
   :alt: Prediction values between feature 4 and feature 6
   :width: 600px  

**Function plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_4_6_real_1.png  
   :alt: Actual interaction values between feature 4 and feature 6 
   :width: 600px  

.. image:: ../_static/experiments/experiment_1/interaction_4_6_real_2.png  
   :alt: Actual interaction values between feature 4 and feature 6
   :width: 600px  


**treemind plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_4_6_treemind.png  
   :alt: treemind interaction values between feature 4 and feature 6
   :width: 600px  

**SHAP plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_4_6_shap.png  
   :alt: SHAP interaction values between feature 4 and feature 6
   :width: 600px  

feature_5 - feature_7  
"""""""""""""""""""""""

**Prediction plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_5_7_pred.png  
   :alt: Prediction values between feature 5 and feature 7
   :width: 600px  

**Function plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_5_7_real_1.png  
   :alt: Actual interaction values between feature 5 and feature 7  
   :width: 600px  

.. image:: ../_static/experiments/experiment_1/interaction_5_7_real_2.png  
   :alt: Actual interaction values between feature 5 and feature 7
   :width: 600px  

**treemind plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_5_7_treemind.png  
   :alt: treemind interaction values between feature 5 and feature 7  
   :width: 600px  

**SHAP plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_5_7_shap.png  
   :alt: SHAP interaction values between feature 5 and feature 7  
   :width: 600px  

feature_6 - feature_8  
"""""""""""""""""""""""

**Prediction plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_6_8_pred.png  
   :alt: Prediction values between feature 6 and feature 8
   :width: 600px  

**Function plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_6_8_real_1.png  
   :alt: Actual interaction values between feature 6 and feature 8
   :width: 600px  

.. image:: ../_static/experiments/experiment_1/interaction_6_8_real_2.png  
   :alt: Actual interaction values between feature 6 and feature 8  
   :width: 600px  

**treemind plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_6_8_treemind.png  
   :alt: treemind interaction values between feature 6 and feature 8  
   :width: 600px  

**SHAP plot:**  

.. image:: ../_static/experiments/experiment_1/interaction_6_8_shap.png  
   :alt: SHAP interaction values between feature 6 and feature 8  
   :width: 600px
