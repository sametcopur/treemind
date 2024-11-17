``treemind.plot``
=================

.. autofunction:: treemind.plot.feature_plot

**Example Usage**

Below is an example of how to use the `feature_plot` function:

.. code-block:: python

    from treemind import Explainer
    from treemind.plot import feature_plot

    # Assume 'model' is a trained lightgbm or xgboost or catboost model object

    # Create an instance of the Explainer
    explainer = Explainer()
    explainer(model)

    # Analyze the specified feature by its index
    df = tree.analyze_feature(22)

    # Plot the feature using a line plot
    feature_plot(df)


**Output**

.. image:: _static/api/feature_plot.png
    :alt: feature_plot example

.. autofunction:: treemind.plot.interaction_plot

**Example Usage**

Below is an example of how to use the `interaction_plot` function:

.. code-block:: python

    from treemind import Explainer
    from treemind.plot import interaction_plot

    # Assume 'model' is a trained lightgbm or xgboost or catboost model object

    # Create an instance of the Explainer
    explainer = Explainer()
    explainer(model)

    # Analyze the specified feature by its index
    df = tree.analyze_feature([22,21])

    # Plot the feature using a line plot
    interaction_plot(df)


**Output**

.. image:: _static/api/interaction_plot.png
    :alt: interaction_plot example

.. autofunction:: treemind.plot.interaction_scatter_plot

**Example Usage**

Below is an example of how to use the `interaction_scatter_plot` function:

.. code-block:: python

    from treemind import Explainer
    from treemind.plot import interaction_scatter_plot

    # Assume 'model' is a trained lightgbm or xgboost or catboost model object

    # Create an instance of the Explainer
    explainer = Explainer()
    explainer(model)

    # Analyze the specified feature by its index
    df = tree.analyze_feature([22,21])

    # Plot the feature using a line plot
    interaction_scatter_plot(X, df, 22, 21)


**Output**

.. image:: _static/api/interaction_scatter_plot.png
    :alt: interaction_scatter_plot example
