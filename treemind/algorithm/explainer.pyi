from typing import (
    Any,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd
from numpy.typing import ArrayLike

class Result:
    """
    Container holding feature–interaction statistics produced by
    :meth:`Explainer.explain`. Acts like a mapping from feature-index tuples
    to per-class :class:`pandas.DataFrame` objects with computed metrics.

    For multi-class models, values are nested dictionaries:

        result[(2, 5)]  ->  {0: df_class0, 1: df_class1, …}

    The ``__getitem__`` method simplifies this by returning a single DataFrame
    merged across classes, with a ``"class"`` column when applicable.

    Notes
    -----
    The content of ``Result`` is intended to be read-only.
    """

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    def __getitem__(self, key: Union[int, Sequence[int]]) -> pd.DataFrame:
        """
        Retrieve interaction statistics for the given feature(s).

        Parameters
        ----------
        key : int or sequence of int
            If ``degree == 1``, a single integer refers to one feature.
            If ``degree >= 2``, a tuple/list must be passed with length equal
            to the interaction degree.

        Returns
        -------
        pandas.DataFrame or None
            Statistics for the specified feature(s), optionally including
            a ``"class"`` column if multi-class.

        Raises
        ------
        ValueError
            If the key length does not match the interaction degree.
        TypeError
            If the key is not an int or sequence of ints.
        """

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
            Sorted by descending importance. Column structure:

            * Degree = 1: ``feature_0``, ``importance`` [+ ``class`` if multi-class]
            * Degree > 1: ``feature_0``, ..., ``feature_{degree-1}``, ``importance`` [+ ``class`` if multi-class]

        Notes
        -----
        Higher ``importance`` implies greater influence over model predictions,
        based on fluctuations under the reference data distribution.
        """

        ...

    def __contains__(self, key: object) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Tuple[int, ...]]: ...
    def keys(self) -> Iterable[Tuple[int, ...]]: ...
    def values(self) -> Iterable[Mapping[int, pd.DataFrame]]: ...
    def items(self) -> Iterable[Tuple[Tuple[int, ...], Mapping[int, pd.DataFrame]]]: ...

class Explainer:
    """
    Extracts interpretable structure from tree models, showing how
    features and feature combinations influence predictions.

    Two main methods are provided:

    - :meth:`explain` — Computes metrics for all interactions of a given degree.
    - :meth:`count_node` — Counts feature appearances in split conditions.

    Usage
    -----
    After initializing, call the object with a trained model:

    >>> explainer = Explainer()
    >>> explainer(model)

    Then use ``explain`` or ``count_node`` as needed.
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

        Raises
        ------
        ValueError
            If the object has not been called with a model or the degree is invalid.
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
            Table sorted by count, with the following columns:

            +-------------------+--------------------------------------------+
            | Column            | Description                                |
            +===================+============================================+
            | ``column1_index`` | Index of the first feature in the group    |
            | ``...``           | ...                                        |
            | ``columnN_index`` | Index of the Nth feature in the group      |
            | ``count``         | Number of times this group appeared        |
            +-------------------+--------------------------------------------+

        Raises
        ------
        ValueError
            If the explainer has not been initialized with a model, or if
            the requested degree is invalid.
        """
