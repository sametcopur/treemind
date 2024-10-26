import pandas as pd
from numpy.typing import ArrayLike
import numpy as np
from typing import Union, Tuple, List, Any

class Explainer:
    """
    The Explainer class provides methods to analyze and interpret a trained model by examining
    feature dependencies, split points, interaction effects, and predicted values. This class
    enables detailed inspection of how individual features and their interactions impact model
    predictions, allowing for a clearer understanding of the model's decision-making process.
    """

    def analyze_interaction(self, main_col: int, sub_col: int) -> pd.DataFrame:
        """
        Analyzes the interaction between two features.

        Parameters
        ----------
        main_col : int
            The column index of the main feature to analyze.

        sub_col : int
            The column index of the sub feature with which to analyze the dependency.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the following columns:

            - `main_feature_lb`: Lower bound for the main feature interval (automatically named by the model).

            - `main_feature_ub`: Upper bound for the main feature interval (automatically named by the model, inclusive).

            - `sub_feature_lb`: Lower bound for the sub feature interval.

            - `sub_feature_ub`: Upper bound for the sub feature interval, inclusive.

            - `value`: A value indicating the interaction effect or dependency strength between the main and sub features within the specified interval combination.

        Notes
        -----
        - The naming of the `main_feature_lb`, `main_feature_ub`, `sub_feature_lb`, and `sub_feature_ub` columns is model-determined. If the column names are unspecified during training, they are auto-assigned based on indices.

        - Each row in the output DataFrame represents a unique combination of intervals between the main and sub features, showing the value associated with the interaction within these intervals.
        """
        ...

    def __call__(self, model: Any) -> None:
        """
        Invokes the Explainer instance with a model to perform analysis.

        Parameters
        ----------
        model : Any
            A trained model instance.

        Returns
        -------
        None
        """
        ...

    def analyze_data(
        self, x: ArrayLike, detailed: bool = False
    ) -> Union[Tuple[np.ndarray, List[np.ndarray], float], Tuple[np.ndarray, float]]:
        """
        Analyzes input data to extract predicted values, feature contributions, split points, and overall raw score.

        Parameters
        ----------
        x : ArrayLike
            Input data for analysis. The data type of `x` should be compatible with the trained model,
            which can accept any type that matches its input requirements. Note that `x` must be
            two-dimensional; single-dimensional arrays are not accepted. If input is intended to
            be row-based, it must have the appropriate shape.

        detailed : bool, optional
            If True, the function returns detailed split points for each feature. If False, only
            basic output is returned. Default is False.

        Returns
        -------
        Union[Tuple[np.ndarray, List[np.ndarray], float], Tuple[np.ndarray, float]]
            The output depends on the `detailed` parameter:

            - If `detailed` is False:
                The function returns a tuple containing:

                - `values` : np.ndarray
                    A single-dimensional array where each element represents the effect (positive or negative) of each feature in `x`. Each index corresponds to a feature column in `x`.

                - `raw_score` : float
                    The mean of the predictions obtained by inputting `x` into the model. This raw score reflects the average output based on `x`.

            - If `detailed` is True:
                The function returns a tuple containing:

                - `values` : np.ndarray
                    A two-dimensional array with shape (n_col, max_split_num_feature). Initially, all values are set to 0. For each feature, the array contains values up to the number of splits for that feature. For example, if a feature has 10 splits and the maximum split count is 30, the first 10 elements will have values, while the rest remain 0. To determine the number of splits for a feature, use `len(split_points[i])`.

                - `split_points` : List[np.ndarray]
                    A list where each element is an array representing the split points for each feature. Each array details the split points where the feature was divided. For example, if a feature splits at 10 different points, the array for that feature contains those 10 split values.

                - `raw_score` : float
                    Similar to the non-detailed case, this represents the mean score of `x` when evaluated by the model.

        """
        ...

    def analyze_feature(self, col: int) -> pd.DataFrame:
        """
        Analyzes a specific feature by calculating the mean, min, and max values
        based on split points across trees for the given column, with upper bounds inclusive.

        Parameters
        ----------
        col : int
            The column index of the feature to analyze.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the following columns:
            - `feature_lb`: Lower bound of the interval for the feature (automatically named by the model).
            - `feature_ub`: Upper bound of the interval for the feature (automatically named by the model, inclusive).
            - `mean`: Average of the data points within the interval according to the model.
            - `min`: Minimum value that data can take in this interval.
            - `max`: Maximum value that data can take in this interval.

        Notes
        -----
        The names of `feature_lb` and `feature_ub` columns are generated by the model and cannot be manually adjusted.
        If no column names are specified during the training phase, they are automatically indexed by the model.
        """
        ...

    def count_node(self, interaction: bool = True) -> pd.DataFrame:
        """
        Counts how often features (or pairs of features if interaction is True) appear in decision splits across the model's trees.

        Parameters
        ----------
        interaction : bool, default True
            If True, counts how often pairs of features appear together in splits.
            If False, counts how often individual features appear in splits.

        Returns
        -------
        pd.DataFrame
            The output depends on the `interaction` parameter:

            - If `interaction` is True:
                The function returns a DataFrame with the following columns:

                - `column1_index` (int): Index of the first feature.

                - `column2_index` (int): Index of the second feature.

                - `count` (int): Number of times the feature pair appears together in splits.

            - If `interaction` is False:
                The function returns a DataFrame with the following columns:

                - `column_index` (int): Index of the feature.

                - `count` (int): Number of times the feature appears in splits.
        """
        ...
