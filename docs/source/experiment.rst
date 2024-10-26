Evaluating treemind's Performance
==================================

This document outlines the process of generating synthetic data and evaluating the treemind model's performance in detecting individual feature effects and interactions. The results show a strong alignment between the actual data and treemind's predictions, demonstrating the model's ability to capture intricate and non-linear relationships within the data.

The visualizations presented in this document are based on the results from the Jupyter Notebook available at the following link:

`treemind Experiment Notebook <https://github.com/sametcopur/treemind/blob/main/examples/experiment.ipynb>`_

Data Transformations and Target Variable Definition
---------------------------------------------------

In this setup, each feature has a unique probability distribution:

1. **feature_0** follows a Gamma distribution with shape = 2 and scale = 2.
2. **feature_1** follows a scaled Beta distribution between 0 and 10.
3. **feature_2** is exponentially distributed with scale = 1.5.
4. **feature_3** is derived from a Chi-square distribution with 3 degrees of freedom.
5. **feature_4** follows a Weibull distribution with a shape parameter of 1.5.
6. **feature_5** is normally distributed with mean = 2 and standard deviation = 1.5.
7. **feature_6** is uniformly distributed between -5 and 5.
8. **feature_7** follows a Laplace distribution centered at 0 with scale = 2.
9. **feature_8** is log-normally distributed with mean = 0 and sigma = 0.5.


**The transformations applied to each feature capture non-linear relationships and diverse functional behaviors.**

.. math::

   \text{transformed_0} = \sin(\text{feature_0}) \cdot e^{-\text{feature_0}/3} + \frac{\text{feature_0}^2}{10}

.. math::

   \text{transformed_1} = 
   \begin{cases}
   \log(1 + \text{feature_1})^2, & \text{if } \text{feature_1} > 5 \\
   \sqrt{\text{feature_1}}, & \text{otherwise}
   \end{cases}

.. math::

   \text{transformed_2} = \tanh(\text{feature_2}) \cdot \cos(\pi \cdot \text{feature_2}) + \frac{\text{feature_2}^3}{20}

.. math::

   \text{transformed_3} = \frac{1}{1 + e^{-\text{feature_3}}} \cdot \log(1 + \text{feature_3})

.. math::

   \text{transformed_4} = \frac{\sinh(\text{feature_4})}{1 + |\text{feature_4}|} + \sqrt{\text{feature_4}}

.. math::

   \text{transformed_5} = 
   \begin{cases}
   \log(1 + |\text{feature_5}|) \cdot \sin(\text{feature_5}), & \text{if } \text{feature_5} > 0 \\
   \log(1 + |\text{feature_5}|) \cdot \cos(\text{feature_5}), & \text{otherwise}
   \end{cases}

.. math::

   \text{transformed_6} = \text{sign}(\text{feature_6}) \cdot \sqrt[3]{|\text{feature_6}|} + \frac{\text{feature_6}^2}{8}

.. math::

   \text{transformed_7} = e^{-|\text{feature_7}|} \cdot \sin\left(\frac{\text{feature_7} \cdot \pi}{4}\right)

.. math::

   \text{transformed_8} = \log(1 + \text{feature_8}) \cdot \tanh\left(\frac{\text{feature_8}}{2}\right)


**These transformations result in feature interactions defined as:**

.. math::

   \text{interaction_0_1} = 
   \begin{cases}
   \text{transformed_0}, & \text{if } \text{transformed_0} > \text{transformed_1} \\
   \text{transformed_1}, & \text{otherwise}
   \end{cases}

.. math::

   \text{interaction_2_3} = \text{transformed_2} \cdot \text{transformed_3} \cdot \text{sign}(\text{transformed_2} + \text{transformed_3})

.. math::

   \text{interaction_4_5} = \max(\text{transformed_4}, \text{transformed_5}) \cdot \min(\text{transformed_4}, \text{transformed_5})

.. math::

   \text{interaction_6_7} = \text{transformed_6} \cdot \sin(\text{transformed_7})

.. math::

   \text{interaction_7_8} = 
   \begin{cases}
   \text{transformed_8}, & \text{if } \text{transformed_7} > 0 \\
   -\text{transformed_8}, & \text{otherwise}
   \end{cases}


**The target variable incorporates these transformations and interactions, weighted by specific coefficients to reflect their influence. Gaussian noise is added to emulate measurement error.**

.. math::

   \text{target} = 0.4 \cdot \text{transformed_0} - 0.6 \cdot \text{transformed_1} + 0.3 \cdot \text{transformed_2} \\
   + 0.5 \cdot \text{transformed_3} - 0.4 \cdot \text{transformed_4} + 0.7 \cdot \text{transformed_5} \\
   - 0.3 \cdot \text{transformed_6} + 0.5 \cdot \text{transformed_7} - 0.4 \cdot \text{transformed_8} \\
   + 0.6 \cdot \text{interaction_0_1} - 0.5 \cdot \text{interaction_2_3} + 0.4 \cdot \text{interaction_4_5} \\
   - 0.3 \cdot \text{interaction_6_7} + 0.5 \cdot \text{interaction_7_8} + \mathcal{N}(0, 0.2)

Comparison of SHAP vs. treemind
-------------------------------

For single-feature plots, the actual target is generated using the function applied to each feature, 
while the treemind predictions use the output from the ``analyze_feature`` function, and SHAP plots use SHAP values for each feature.

For two-feature interaction plots, the actual target is constructed row-by-row as a function of ``feature1``, ``feature2``, 
and their interaction, represented as ``(a * transformed1 + b * transformed2 + c * interaction1_2)``. The colormap indicates 
the effect of the target. Similarly, treemind uses the output from ``analyze_interaction``, while SHAP plots use SHAP interaction 
values for the corresponding features.


Feature Analysis
^^^^^^^^^^^^^^^^^

feature_0
""""""""""

**Function plot:**

.. image:: _static/experiment/feature_0_real.png
   :alt: Contribution of feature 0
   :width: 600px

**treemind plot:**

.. image:: _static/experiment/feature_0_treemind.png
   :alt: treemind's extracted values for feature 0
   :width: 600px

**SHAP plot:**

.. image:: _static/experiment/feature_0_shap.png
   :alt: SHAP values for feature 0
   :width: 600px

feature_5  
""""""""""

**Function plot:** 

.. image:: _static/experiment/feature_5_real.png  
   :alt: Contribution of feature 5  
   :width: 600px  

**treemind plot:**  

.. image:: _static/experiment/feature_5_treemind.png  
   :alt: treemind's extracted values for feature 5  
   :width: 600px  

**SHAP plot:**  

.. image:: _static/experiment/feature_5_shap.png  
   :alt: SHAP values for feature 5  
   :width: 600px  


feature_6  
""""""""""

**Function plot:**  

.. image:: _static/experiment/feature_6_real.png  
   :alt: Contribution of feature 6  
   :width: 600px  

**treemind plot:**  

.. image:: _static/experiment/feature_6_treemind.png  
   :alt: treemind's extracted values for feature 6  
   :width: 600px  

**SHAP plot:**  

.. image:: _static/experiment/feature_6_shap.png  
   :alt: SHAP values for feature 6  
   :width: 600px  


feature_7  
""""""""""

**Function plot:**  

.. image:: _static/experiment/feature_7_real.png  
   :alt: Contribution of feature 7  
   :width: 600px  

**treemind plot:**  

.. image:: _static/experiment/feature_7_treemind.png  
   :alt: treemind values for feature 7  
   :width: 600px 

**SHAP plot:**  

.. image:: _static/experiment/feature_7_shap.png  
   :alt: SHAP values for feature 7  
   :width: 600px  


Interaction Analysis
^^^^^^^^^^^^^^^^^^^^^

feature_0 - feature_1 
"""""""""""""""""""""

**Function plot:**  

.. image:: _static/experiment/interaction_0_1_real.png  
   :alt: Actual interaction values between feature 0 and feature 1
   :width: 600px  

**treemind plot:**  

.. image:: _static/experiment/interaction_0_1_treemind.png  
   :alt: treemind interaction values between feature 0 and feature 1
   :width: 600px  

**SHAP plot:**  

.. image:: _static/experiment/interaction_0_1_shap.png  
   :alt: SHAP interaction values between feature 0 and feature 1
   :width: 600px  

feature_2 - feature_3  
"""""""""""""""""""""""

**Function plot:**  

.. image:: _static/experiment/interaction_2_3_real.png  
   :alt: Actual interaction values between feature 2 and feature 3  
   :width: 600px  

**treemind plot:**  

.. image:: _static/experiment/interaction_2_3_treemind.png  
   :alt: treemind interaction values between feature 2 and feature 3  
   :width: 600px  

**SHAP plot:**  

.. image:: _static/experiment/interaction_2_3_shap.png  
   :alt: SHAP interaction values between feature 2 and feature 3  
   :width: 600px  

feature_4 - feature_5  
"""""""""""""""""""""""

**Function plot:**  

.. image:: _static/experiment/interaction_4_5_real.png  
   :alt: Actual interaction values between feature 4 and feature 5  
   :width: 600px  

**treemind plot:**  

.. image:: _static/experiment/interaction_4_5_treemind.png  
   :alt: treemind interaction values between feature 4 and feature 5  
   :width: 600px  

**SHAP plot:**  

.. image:: _static/experiment/interaction_4_5_shap.png  
   :alt: SHAP interaction values between feature 4 and feature 5  
   :width: 600px  

feature_6 - feature_7  
"""""""""""""""""""""""

**Function plot:**  

.. image:: _static/experiment/interaction_6_7_real.png  
   :alt: Actual interaction values between feature 6 and feature 7  
   :width: 600px  

**treemind plot:**  

.. image:: _static/experiment/interaction_6_7_treemind.png  
   :alt: treemind interaction values between feature 6 and feature 7  
   :width: 600px  

**SHAP plot:**  

.. image:: _static/experiment/interaction_6_7_shap.png  
   :alt: SHAP interaction values between feature 6 and feature 7  
   :width: 600px  

feature_7 - feature_8  
"""""""""""""""""""""""

**Function plot:**  

.. image:: _static/experiment/interaction_7_8_real.png  
   :alt: Actual interaction values between feature 7 and feature 8  
   :width: 600px  

**treemind plot:**  

.. image:: _static/experiment/interaction_7_8_treemind.png  
   :alt: treemind interaction values between feature 7 and feature 8  
   :width: 600px  

**SHAP plot:**  

.. image:: _static/experiment/interaction_7_8_shap.png  
   :alt: SHAP interaction values between feature 7 and feature 8  
   :width: 600px
