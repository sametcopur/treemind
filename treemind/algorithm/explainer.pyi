from typing import (
    Any,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Dict,
    Union,
)

import pandas as pd
from numpy.typing import ArrayLike

class Result:
    """
    Container holding feature–interaction statistics produced by
    :meth:`Explainer.explain`. Acts like a mapping from feature-index tuples
    to per-class :class:`pandas.DataFrame` objects with computed metrics.

    The ``__getitem__`` method simplifies this by returning a single DataFrame
    merged across classes, with a ``"class"`` column when applicable.

    Notes
    -----
    The content of ``Result`` is intended to be read-only.
    """

    degree: int
    n_classes: int
    feature_names: list[str]
    model_type: str

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    
    def __getitem__(self, key: Union[int, Sequence[int]]) -> pd.DataFrame: ...
    def __contains__(self, key: object) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Tuple[int, ...]]: ...

    @property
    def data(self) -> Dict[Tuple[int, ...], Dict[int, pd.DataFrame]]:
        """
        Internal dictionary mapping feature-index tuples to per-class DataFrames.

        Returns
        -------
        dict
            The full internal representation of interaction data.
        """
        ...

    def importance(self, combine_classes: bool = False) -> pd.DataFrame:
        """
        Calculate the mean absolute contribution metric (``I_abs``) for each
        feature or interaction.

        The importance score for a group is:

        .. math::

            I_{\\text{abs}} = \\frac{\\sum_i \\left| E[F\\mid i]-\\mu \\right| \\cdot \\text{count}_i}{\\sum_i \\text{count}_i}

        Parameters
        ----------
        combine_classes : bool, default=False
            If True, aggregates per-class ``I_abs`` into a single weighted
            value per group. If False, returns a row per class.

        Returns
        -------
        pandas.DataFrame
            A DataFrame listing feature or interaction importances (``I_abs``),
            sorted by descending importance.
        """
        ...

class Explainer:
    """
    Extracts interpretable structure from tree models, showing how
    features and feature combinations influence predictions.
    """

    def __init__(self, model: Any) -> None:
        """
        Parameters
        ----------
        model : object
            A trained tree-based model to be analyzed.
        """
        ...

    def __repr__(self) -> str: ...
    def explain(
        self,
        degree: int,
        *,
        back_data: Optional[ArrayLike] = None,
    ) -> Result:
        """
        Compute interaction metrics for feature groups of a specified degree.

        Parameters
        ----------
        degree : int
            Interaction order: 1 for main effects, 2 for pairs, etc.
        back_data : array-like, optional
            Optional dataset used to re-weight statistics for baseline-specific
            explanations.

        Returns
        -------
        Result
            A mapping from feature index tuples to per-class DataFrames.
        """

    def count_node(self, degree: int = 2) -> pd.DataFrame:
        """
        Count how often feature groups appear in tree split rules.

        Parameters
        ----------
        degree : int, default=2
            Number of features in each group to count.

        Returns
        -------
        pandas.DataFrame
            A DataFrame listing feature groups of the given degree and how
            often they appear in tree split rules, sorted by descending count.
        """