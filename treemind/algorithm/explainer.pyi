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
        
    def analyze_feature(
        self,
        columns: Union[int, List[int]],
        back_data: Optional[ArrayLike] = None
    ) -> pd.DataFrame:
        """
        Analyzes interactions between multiple features based on the model's decision rules,
        returning a DataFrame with interaction metrics.

        Parameters
        ----------
        columns : int or list[int]
            Column indice or list of column indices representing features to analyze for interactions.
            
        back_data : Optional[object], default=None
            Optional data used to update leaf counts in the decision trees. If provided, this
            data is used to refine the interaction analysis.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the interaction analysis metrics:
            - Upper bound columns for each feature in `columns`
            - Mean interaction values, standard deviations, and interaction counts for each
            feature pair.
        """
        ...

    def analyze_data(
        self, x: ArrayLike, back_data: Optional[ArrayLike] = None
    ) -> np.ndarray:
        """
        Analyzes input data to extract row and column-based impact values, with an option to use
        baseline data for feature impact calculation.

        Parameters
        ----------
        x : ArrayLike
            Input data for analysis. The data type of `x` should be compatible with the trained model,
            which can accept any type that matches its input requirements. Note that `x` must be
            two-dimensional; single-dimensional arrays are not accepted. If input is intended to
            be row-based, it must have the appropriate shape.
            
        back_data : ArrayLike, optional
            Baseline data used to calculate impact values. When provided, each feature's effect is
            computed as the deviation from this baseline value, similar to SHAP analysis. If `None`, 
            the function will use the model’s expected output as the reference for impact calculations.

        Returns
        -------
        np.ndarray
            A two-dimensional array where each element represents the impact of a feature in `x` on a
            specific row, based on either the provided baseline (`back_data`) or the model's output 
            as the reference if `back_data` is `None`. The array shape corresponds to (n_rows, n_features), 
            where each row gives the per-feature impact for a specific instance in `x`, enabling row 
            and column-based impact analysis.
            
        """
        ...



    def count_node(self, order: int = 2) -> pd.DataFrame:
        """
        Counts how often combinations of features appear in decision splits across the model's trees.

        Parameters
        ----------
        order : int, default 2
            Specifies the number of features in each combination to count.

            - If `order=1`, counts how often individual features appear in splits.
            - If `order=2`, counts how often pairs of features appear together in splits.
            - For higher values of `order`, counts combinations of the specified number of features.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the counts of feature combinations appearing in splits, with the following columns:

            - `column1_index`, `column2_index`, ..., `columnN_index` (int): Indices of the features in the combination,
              where `N` equals the `order` parameter.

            - `count` (int): Number of times the feature combination appears in splits.

        """
        ...


