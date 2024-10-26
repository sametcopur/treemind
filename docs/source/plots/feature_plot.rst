.. autofunction:: treemind.plot.feature_plot

**Example Usage**

Below is an example of how to use the `feature_plot` function:

.. code-block:: python

    from treemind import Explainer
    from treemind.plot import feature_plot

    # Assume 'model' is a trained LightGBM or XGBoost model object

    # Create an instance of the Explainer
    explainer = Explainer()
    explainer(model)

    # Analyze the specified feature by its index
    df = tree.analyze_feature(22)

    # Plot the feature using a line plot
    feature_plot(df)


**Output**

.. image:: ../_static/api/feature_plot.png
    :alt: feature_plot example
