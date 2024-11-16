Experiment 4
============

.. code-block:: python

   target = (
      (transformed_0 - transformed_1) ** 2
      + (transformed_2 - transformed_3) ** 2
      + np.random.normal(loc=0, scale=0.2, size=n_samples)
   )

Interaction Analysis
^^^^^^^^^^^^^^^^^^^^^

feature_0 - feature_1 
"""""""""""""""""""""

**Prediction plot:**  

.. image:: ../_static/experiments/experiment_4/interaction_0_1_pred.png  
   :alt: Prediction values between feature 0 and feature 1
   :width: 600px  

**Function plot:**  

.. image:: ../_static/experiments/experiment_4/interaction_0_1_real_1.png  
   :alt: Actual interaction values between feature 0 and feature 1
   :width: 600px  

.. image:: ../_static/experiments/experiment_4/interaction_0_1_real_2.png  
   :alt: Actual interaction values between feature 0 and feature 1
   :width: 600px  

**treemind plot:**  

.. image:: ../_static/experiments/experiment_4/interaction_0_1_treemind.png  
   :alt: treemind interaction values between feature 0 and feature 1
   :width: 600px  

**SHAP plot:**  

.. image:: ../_static/experiments/experiment_4/interaction_0_1_shap.png  
   :alt: SHAP interaction values between feature 0 and feature 1
   :width: 600px  

feature_2 - feature_3
"""""""""""""""""""""

**Prediction plot:**  

.. image:: ../_static/experiments/experiment_4/interaction_2_3_pred.png  
   :alt: Prediction values between feature 2 and feature 3
   :width: 600px  

**Function plot:**  

.. image:: ../_static/experiments/experiment_4/interaction_2_3_real_1.png  
   :alt: Actual interaction values between feature 2 and feature 3
   :width: 600px  

.. image:: ../_static/experiments/experiment_4/interaction_2_3_real_2.png  
   :alt: Actual interaction values between feature 2 and feature 3
   :width: 600px  


**treemind plot:**  

.. image:: ../_static/experiments/experiment_4/interaction_2_3_treemind.png  
   :alt: treemind interaction values between feature 2 and feature 3
   :width: 600px  

**SHAP plot:**  

.. image:: ../_static/experiments/experiment_4/interaction_2_3_shap.png  
   :alt: SHAP interaction values between feature 2 and feature 3
   :width: 600px  