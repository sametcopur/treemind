import pandas as pd
from numpy.typing import ArrayLike
import numpy as np
from typing import List, Any, Optional, Union

class Explainer:
    """
    The Explainer class provides methods to analyze and interpret a trained model by examining
    feature dependencies, split points, interaction effects, and predicted values. This class
    enables detailed inspection of how individual features and their interactions impact model
    predictions, allowing for a clearer understanding of the model's decision-making process.
    """

    def __call__(self, model: Any) -> None:
        """
        The Explainer class provides methods to analyze and interpret trained models by examining
        feature dependencies, split points, interaction effects, and predicted values. This class
        enables detailed inspection of how individual features and their interactions impact model
        predictions, offering insights into the model's decision-making process.

        Parameters
        ----------
        model : Any
            A trained model instance.

        Returns
        -------
        None
        """
        ...
    def analyze_feature(
        self, columns: Union[int, List[int]], back_data: Optional[ArrayLike] = None
    ) -> pd.DataFrame:
        """
        Analyzes feature interactions based on the model's decision rules and computes metrics
        that quantify their combined influence on predictions.

        Parameters
        ----------
        columns : int or list[int]
            The index or list of indices of the features to analyze for interactions. Each index
            corresponds to a feature in the input data.

        back_data : Optional[ArrayLike], default=None
            Optional data for updating the tree's leaf counts dynamically. This data allows for
            re-calculating interaction metrics based on new or baseline data.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the interaction analysis results with the following columns:
            
            - `feature_X_lower_bound`, `feature_X_upper_bound` (float): Lower and upper bounds
              for each analyzed feature.
            - `interaction_value` (float): The calculated metric representing the combined influence
              of the features.
            - `std_dev` (float): The standard deviation of the interaction value.
            - `average_leaf_count` (float): The average count of data points across relevant tree leaves.
        """
        ...

    def count_node(self, order: int = 2) -> pd.DataFrame:
        """
        Counts the frequency of feature combinations used in decision splits across all trees
        in the model.

        Parameters
        ----------
        order : int, default=2
            The number of features in each combination to count:
            
            - `order=1`: Counts how often individual features are used in splits.
            - `order=2`: Counts how often pairs of features appear together in splits.
            - `order=N`: Counts combinations of `N` features.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the following columns:
            
            - `feature_1_index`, `feature_2_index`, ..., `feature_N_index` (int): Indices of the
              features in the combination, where `N` equals the `order` parameter.
            - `count` (int): The number of times this feature combination appears in the splits
              across all trees.
        """
        ...