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
    :pymeth:`Explainer.explain`.  The object behaves like a mapping whose
    **keys** are feature-index tuples and whose **values** are per-class
    :class:`pandas.DataFrame` objects with the calculated metrics.

    In multi-class models the underlying dictionary is two-level:

    ``result[(2, 5)]  ->  {0: df_class0, 1: df_class1, …}``

    For convenience ``Result`` implements ``__getitem__`` so that
    requesting a key returns a *single* DataFrame, concatenating the
    class-wise frames and inserting a ``"class"`` column when necessary.

    Notes
    -----
    ``Result`` is read-only from a public API perspective; downstream
    code should treat its content as immutable.

    Examples
    --------
    >>> res = explainer.explain(degree=2)
    >>> (2, 5) in res          # membership test
    True
    >>> res[2, 5].head()       # statistics for the interaction of feature 2 & 5
    """

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    def __getitem__(self, key: Union[int, Sequence[int]]) -> Optional[pd.DataFrame]:
        """
        Retrieve interaction statistics for the given feature(s).

        Parameters
        ----------
        key : int or sequence of int
            * **Degree = 1** – a single integer is interpreted as the
              index of a *single* feature.
            * **Degree ≥ 2** – a list/tuple whose length exactly matches
              the degree of interactions stored in this ``Result``.

        Returns
        -------
        pandas.DataFrame or None
            * A DataFrame with the requested statistics.

        Raises
        ------
        ValueError
            If the length of *key* does not match the stored degree.
        TypeError
            If *key* is neither ``int`` nor a sequence of ``int``.
        """

    def importance(self, combine_classes: bool = False) -> pd.DataFrame:
        """
        Compute the *mean absolute contribution* metric (``I_abs``) for
        every stored feature or feature-interaction group.

        The score for a given group is defined as

        .. math::

            I_{\\text{abs}} = \\frac{\\sum_i \\left| E[F\\mid i]-\\mu \\right|
                               \\;\\cdot\\; \\text{count}_i}
                              {\\sum_i \\text{count}_i}

        where ``E[F|i]`` is the interval-conditioned expectation,
        ``count_i`` the corresponding sample count and
        :math:`\\mu = E[F]` the global expectation.

        Parameters
        ----------
        combine_classes : bool, default ``True``
            * **True** – for multi-class models, class-specific
              ``I_abs`` values are aggregated into **one** number per
              feature group, weighted by the sample count of each class.
            * **False** – returns a separate row *per class*; the
              resulting :class:`~pandas.DataFrame` includes an
              additional ``"class"`` column.

        Returns
        -------
        pandas.DataFrame
            Sorted by descending importance.

            * **Degree = 1** columns: ``"feature_0"``, ``"importance"``
              plus optional ``"class"``.
            * **Degree > 1** columns: ``"feature_0"``, …,
              ``"feature_{degree-1}"``, ``"importance"``
              (and optionally ``"class"``).

        Notes
        -----
        A larger ``importance`` indicates that fluctuations in this
        feature (or interaction) explain a greater share of variation in
        model predictions under the distribution represented by
        *back-data*.
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
    Extracts human-readable structure from tree models and quantifies how individual
    features—and their combinations—shape the ensemble’s predictions.

    Two complementary analysis pathways are available:

    * :pymeth:`explain` – enumerates every interaction of a given
      *degree* (1 → marginal effects, 2 → pairwise, …) and returns a
      :class:`Result` object with point estimates, uncertainty, and leaf
      counts.
    * :pymeth:`count_node` – counts how frequently specific features (or
      feature groups) occur in tree split rules.

    The class must be *called* first with a trained model to parse its
    trees and cache internal state::

        explainer = Explainer()
        explainer(tree_model)

    Thereafter any number of analyses can be executed without
    re-parsing the booster.
    """

    def __init__(self, model: Any) -> None:
        """
        Parameters
        ----------
        model : A **trained** tree model.
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
        Quantify the joint influence of *degree*-sized feature groups on
        predicted outcomes.

        Parameters
        ----------
        degree : int
            Interaction order to inspect.  ``degree=1`` yields
            per-feature main effects; ``degree=2`` yields pairwise
            interactions, etc.
        back_data : ArrayLike, optional
            Reference dataset used to *re-weight* leaf statistics.  When
            supplied, each tree’s leaf counts are recomputed with
            ``back_data`` before metrics are aggregated, enabling
            conditional or baseline-specific explanations.

        Returns
        -------
        Result
            Mapping-like object whose keys are feature-index tuples of
            length *degree*.

        Raises
        ------
        ValueError
            If ``Explainer.__call__`` has not been invoked or *degree*
            is out of bounds.
        """

    def count_node(self, degree: int = 2) -> pd.DataFrame:
        """
        Enumerate how often features (or feature combinations) appear in
        split conditions across **all** trees.

        Parameters
        ----------
        degree : int, default=2
            Size of the feature group to count.

        Returns
        -------
        pandas.DataFrame
            Sorted table with columns

            ================  =========================================
            ``column1_index``  Index of the **1st** feature in group
            «…»
            ``columnN_index``  Index of the **Nth** feature in group
            ``count``          Number of split rules containing *all*
                               listed features
            ================  =========================================

        Raises
        ------
        ValueError
            If ``Explainer.__call__`` has not been invoked or *degree*
            is out of bounds.
        """
