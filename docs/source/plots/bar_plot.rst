.. autofunction:: treemind.plot.bar_plot

**Example Usage**

Below is an example of how to use the `bar_plot` function:

.. code-block:: python

    from treemind import Explainer
    from treemind.plot import bar_plot

    # Assume 'model' is a trained LightGBM or XGBoost or CatBoost model object

    # Create an explainer instance
    explainer = Explainer()
    explainer(model)

    # Analyze the entire dataset
    values = explainer.analyze_data(X)

    # Alternatively, analyze a single row
    # values, raw_score = explainer.analyze_data(X.iloc[10, :])

    # Plot the feature importance using a bar plot
    bar_plot(values, raw_score, columns=X_train.columns)

**Output**

.. image:: ../_static/api/bar_plot.png
    :alt: bar_plot example
