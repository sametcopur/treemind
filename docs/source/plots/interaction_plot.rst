.. autofunction:: treemind.plot.interaction_plot

**Example Usage**

Below is an example of how to use the `interaction_plot` function:

.. code-block:: python

    from treemind import Explainer
    from treemind.plot import interaction_plot

    # Assume 'model' is a trained LightGBM or XGBoost model object

    # Create an instance of the Explainer
    explainer = Explainer()
    explainer(model)

    # Analyze the specified feature by its index
    df = tree.analyze_feature(22)

    # Plot the feature using a line plot
    interaction_plot(df)


**Output**

.. image:: ../_static/api/interaction_plot.png
    :alt: interaction_plot example
