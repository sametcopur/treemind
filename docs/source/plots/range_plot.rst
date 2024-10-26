.. autofunction:: treemind.plot.range_plot

**Example Usage**

Below is an example of how to use the `range_plot` function:

.. code-block:: python

    from treemind import Explainer
    from treemind.plot import feature_plot

    # Assume 'model' is a trained LightGBM or XGBoost model object

    # Create an instance of the Explainer
    explainer = Explainer()
    explainer(model)

    # Analyze the specified feature by its index
    values, split_points, raw_score = explainer.analyze_row(X_train, detailed=True)

    # Plot the feature using a line plot
    range_plot(values, split_points, raw_score)


**Output**

.. image:: ../_static/api/range_plot.png
    :alt: feature_plot example
