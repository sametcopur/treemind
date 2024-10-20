import pandas as pd
from numpy.typing import ArrayLike
import numpy as np
from typing import Union, Tuple, List, Any


class Explainer:
    """
    The Explainer class provides methods to analyze a trained model by examining feature dependencies,
    split points, and predicted values. It offers functionality to analyze the relationship between
    features, examine the impact of individual features on predictions, and interpret the model's
    decision-making process.
    """

    def analyze_interaction(self, main_col: int, sub_col: int) -> pd.DataFrame:
        """
        Analyzes the dependency between two features

        Parameters
        ----------
        main_col : int
            The column index of the main feature.
        sub_col : int
            The column index of the sub feature.

        Returns
        -------
        pd.DataFrame
            A DataFrame with sub feature split points, main feature split points, and the corresponding values.
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
        self, x: ArrayLike, detailed: bool = True
    ) -> Union[Tuple[np.ndarray, List[np.ndarray], float], Tuple[np.ndarray, float]]:
        """
        Analyzes input data to extract predicted values and split points.

        Parameters
        ----------
        x : np.ndarray
            Input data for which predictions are made.
        detailed : bool, optional
            If True, returns detailed split points for each feature. Default is True.

        Returns
        -------
        Union[Tuple[np.ndarray, List[np.ndarray], float], Tuple[np.ndarray, float]]
            If detailed is True, returns a tuple of values array, list of split points, and raw score.
            Otherwise, returns a tuple of values array and raw score.
        """

        ...
    
    def analyze_feature(self, col: int) -> pd.DataFrame:
        """
        Analyzes a specific feature by calculating the mean, min, and max values
        based on split points across trees for the given column.

        Parameters
        ----------
        col : int
            The column index of the feature to analyze.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the split points (main_point), mean, min, and max values
            for the specified feature.
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
            If interaction is True:
                A DataFrame with the following columns:
                - `column1_index` (int): Index of the first feature.
                - `column2_index` (int): Index of the second feature.
                - `count` (int): Number of times the feature pair appears together in splits.
            If interaction is False:
                A DataFrame with the following columns:
                - `column_index` (int): Index of the feature.
                - `count` (int): Number of times the feature appears in splits.
        """
        ...

