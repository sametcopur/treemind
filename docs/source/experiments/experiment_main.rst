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


Data Generation Process
-------------------------

The datasets used in this analysis were synthetically generated, adhering to specific distributions. Features were named as ``feature_0, feature_1, ..., feature_n`` to maintain clarity.

Subsequently, transformations, denoted as ``transform_0, transform_1, ..., transform_n``, were applied to these features. These transformations included operations such as conditional statements (e.g., ``where`` conditions), trigonometric functions like ``sin``, or more complex functions. Each transformation utilized only its corresponding feature (e.g., ``transform_0`` was applied solely to ``feature_0``).

Finally, feature interactions, such as ``interaction_0_1``, were constructed using combinations of transformations, e.g., ``interaction_0_1 = function(transform_0, transform_1)``. These interactions could be continuous or discrete depending on the chosen function.

The target variable was generated using the formula:

.. math::

   \text{target} = (a \cdot \text{transformed_0}) + (b \cdot \text{transformed_1}) + (c \cdot \text{interaction_0_1})

where ``a``, ``b``, and ``c`` are coefficients. A regression problem was then set up to predict this target.

For single-feature analyses, simple line plots and scatter plots were employed. The x-axis represented feature values, while the y-axis varied depending on the context. For SHAP values, the y-axis displayed the SHAP values; for Treemind, it showed the expected values. In cases involving the true values, the interaction effect of the analyzed feature on the target was highlighted. The values were calculated such that, for example, if ``feature_0`` alone influenced the target alongside ``interaction_0_1``, the expression:

.. math::

   \text{feature_0 + interaction_0_1} - \text{mean(feature_0 + interaction_0_1)}

was evaluated to remove the mean offset.

For pairwise analyses, scatter plots were used with the x and y axes representing feature values and a color bar for the values. Additionally, ``prediction - mean(prediction)`` was considered in pairwise analyses to detect interactions clearly visible in predictions, ensuring the algorithm was not portrayed as overly successful. This approach allowed users to assess interactions critically.

A notable challenge in pairwise interactions is distinguishing between capturing only ``interaction_0_1`` versus capturing the combined effect of ``transformed_0``, ``transformed_1``, and ``interaction_0_1``. This distinction was explicitly highlighted in the analysis.

Furthermore, two additional tests were designed to evaluate specific cases. Details of these tests can be found in the experiments section of the provided GitHub repository. 

`treemind experiments <https://github.com/sametcopur/treemind/blob/main/examples/>`_