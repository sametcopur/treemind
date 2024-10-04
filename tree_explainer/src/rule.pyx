from libcpp.vector cimport vector
from libcpp.algorithm cimport sort, unique
from libc.math cimport INFINITY

from cython cimport boundscheck, wraparound
from .rule cimport Rule

import numpy as np

@boundscheck(False)
@wraparound(False)
cdef Rule create_rule(int len_col, int tree_index, int leaf_index):
    """
    Initializes a new Rule struct with the given parameters.

    Parameters
    ----------
    len_col : int
        The number of features.
    tree_index : int
        Index of the tree in the ensemble.
    leaf_index : int
        Index of the leaf in the tree.

    Returns
    -------
    Rule
        The initialized Rule struct.
    """
    cdef Rule rule
    rule.len_col = len_col
    rule.tree_index = tree_index
    rule.leaf_index = leaf_index
    rule.lbs = vector[double](len_col, -INFINITY)
    rule.ubs = vector[double](len_col, INFINITY)
    rule.value = np.nan
    return rule

@boundscheck(False)
@wraparound(False)
cdef void update_rule(Rule* rule, int index, double lb, double ub):
    """
    Updates the lower and upper bounds for a feature at the specified index.

    Parameters
    ----------
    rule : Rule*
        The Rule struct to update.
    index : int
        The index of the feature to update.
    ub : float
        The new upper bound for the feature.
    lb : float
        The new lower bound for the feature.
    """
    rule.lbs[index] = max(rule.lbs[index], lb)
    rule.ubs[index] = min(rule.ubs[index], ub)

@boundscheck(False)
@wraparound(False)
cdef inline bint check_rule(Rule* rule, vector[int] feature_indices) :
    """
    Checks whether the rule has been fully defined for the given feature indices.

    Parameters
    ----------
    rule : Rule*
        The Rule struct to check.
    feature_indices : int[:]
        A list of feature indices to check.

    Returns
    -------
    bint
        True if all feature bounds have been specified (i.e., not infinite), False otherwise.
    """
    cdef int idx
    for idx in feature_indices:
        if (rule.lbs[idx] == -INFINITY) and (rule.ubs[idx] == INFINITY):
            return 0
    return 1

@boundscheck(False)
@wraparound(False)
cdef inline bint check_value(Rule* rule, int i, double value) noexcept nogil:
    """
    Checks if the given value lies within the bounds for the specified feature index.

    Parameters
    ----------
    rule : Rule*
        The Rule struct to check.
    i : int
        The feature index to check.
    value : double
        The value to check against the feature's bounds.

    Returns
    -------
    bint
        True if the value lies within the bounds, False otherwise.
    """
    return rule.lbs[i] < value <= rule.ubs[i]


@boundscheck(False)
@wraparound(False)
cdef int compare_rules(const Rule& a, const Rule& b):
    return a.leaf_index < b.leaf_index

@boundscheck(False)
@wraparound(False)
cdef vector[vector[Rule]] filter_trees(vector[vector[Rule]] trees, int main_col, int sub_col = -1):
    """
    Filters rules across all trees based on the specified main and optional sub-column.

    Parameters
    ----------
    main_col : int
        The primary column index to filter by.
    sub_col : Optional[int], optional
        The secondary column index to filter by, by default None.

    Returns
    -------
    list
        A list of lists, where each inner list contains rules that match the given conditions.
    """

    cdef Rule rule
    cdef vector[vector[Rule]] filtered_trees
    cdef vector[Rule] filtered_rules = vector[Rule]()
    cdef vector[Rule] tree
    cdef vector[int] check_rules = vector[int]() 

    # Convert Python int to C++ int
    cdef int c_main_col = <int>main_col
    cdef int c_sub_col = <int>sub_col
    
    check_rules.push_back(c_main_col)

    if c_sub_col != -1:
        check_rules.push_back(c_sub_col)

    for tree in trees:
        filtered_rules.clear()
        for rule in tree:
            if check_rule(&rule, check_rules):
                filtered_rules.push_back(rule)

        if filtered_rules.size() >= 1:
            filtered_trees.push_back(filtered_rules)

    return filtered_trees


@boundscheck(False)
@wraparound(False)
cdef vector[double] get_split_point(vector[vector[Rule]] trees, int col):
    """
    Retrieves the unique split points for a specific column across all trees, sorted in ascending order.

    Parameters
    ----------
    trees : vector[vector[Rule]]
        The list of trees, where each tree is a vector of Rule objects.
    col : int
        The column index for which to retrieve split points.

    Returns
    -------
    vector[double]
        A vector of unique split points for the specified column, excluding the smallest one.
    """

    cdef vector[double] points
    cdef vector[Rule] tree
    cdef Rule rule
    cdef vector[double].iterator it
    cdef vector[double] result

    for tree in trees:
        for rule in tree:
            points.push_back(rule.ubs[col])
            points.push_back(rule.lbs[col])

    sort(points.begin(), points.end())
    it = unique(points.begin(), points.end())
    
    points.resize(it - points.begin())

    if len(points) > 1:
        points.erase(points.begin())
        
    return points