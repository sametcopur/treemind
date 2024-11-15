Evaluating treemind's Performance
==================================
treemind represents an experimental approach to feature and feature interaction analysis, developed through practical observations rather 
than theoretical foundations. Our extensive testing reveals interesting patterns in its behavior that are worth examining in detail.


The algorithm's performance exhibits considerable variability—even with similarly structured synthetic data, Treemind can produce both 
meaningful and less interpretable results. This variability primarily stems from its dependence on the underlying tree structure of the 
model. In some cases, it successfully isolates feature interactions, while in others, it may capture collective effects of features and  
on the other hand it may isolate nothing. 

Notably, in single-feature analysis, Treemind generally aligns well with results obtained from widely-used libraries like SHAP, demonstrating 
consistency in this aspect. However, when it comes to analyzing feature interactions, its behavior appears more experimental, as it does not 
follow a standardized framework. Instead, it reflects the nuances of the tree-based model's partitioning, which can be both an advantage and 
a challenge depending on the use case.

One notable characteristic of *Treemind* is its computational efficiency. When analyzing single-feature effects, it typically produces quick 
results—a trait it shares with most frameworks, as single-feature analysis is generally computationally lightweight across the board. However,
when it comes to feature interactions, many frameworks tend to slow down significantly due to the complexity of capturing these relationships.

treemind, in contrast, stands out with its exceptional speed in this area. It can analyze highly complex interactions in just a few seconds,
making it an excellent tool for exploratory analysis where rapid insights into feature relationships are valuable. This remarkable efficiency 
provides a practical edge, especially in scenarios requiring iterative experimentation or large-scale datasets.

It's important to note that treemind makes no claims of mathematical guarantees or absolute consistency. Instead, it evolved from practical 
applications and empirical observations, offering a pragmatic, if sometimes uncertain, approach to understanding feature relationships in 
tree-based models.

The visualizations presented in this document are based on the results of experiments available at the following link:

`treemind experiments <https://github.com/sametcopur/treemind/blob/main/examples/>`_

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
