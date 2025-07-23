from libcpp.vector cimport vector
from libcpp.algorithm cimport sort, unique
from libc.math cimport INFINITY

from .rule cimport Rule
from .xgb cimport convert_d_matrix, xgb_leaf_correction

cimport numpy as cnp
import numpy as np

from collections import Counter

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
    cdef int i

    rule.len_col = len_col
    rule.tree_index = tree_index
    rule.leaf_index = leaf_index
    rule.lbs = vector[float](len_col, -INFINITY)
    rule.ubs = vector[float](len_col, INFINITY) 
    rule.value = np.nan
    rule.count = -1
    rule.cats = vector[vector[bint]]()
    rule.cat_flags = vector[bint]()
    return rule

cdef void update_cats_for_rule(Rule* rule, const vector[vector[bint]]& cats):
    """
    Copies the active categorical mask constraints into the Rule struct and
    prints them (debug).

    Parameters
    ----------
    rule : Rule*
        The Rule struct being constructed for a leaf node.
    cats : vector[vector[bint]]&
        Temporary categorical mask collected during tree traversal.
        Each cats[i] is a boolean mask indicating allowed category values for feature i.
    """
    cdef size_t i, j
    cdef bint all_one
    rule.cats.resize(cats.size())
    rule.cat_flags.resize(cats.size(), 0)

    for i in range(cats.size()):
        rule.cats[i] = cats[i]

        all_one = True
        for j in range(cats[i].size()):
            if cats[i][j] == 0:
                all_one = False
                break
        rule.cat_flags[i] = 0 if all_one else 1

cdef void update_rule(Rule* rule, int index, float lb, float ub):
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


cdef inline bint check_rule(const Rule* rule, vector[int] feature_indices)  noexcept nogil:
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

cdef inline bint check_value(const Rule* rule, int i, float value) noexcept nogil:
    """
    Checks if the given value lies within the bounds for the specified feature index.

    Parameters
    ----------
    rule : Rule*
        The Rule struct to check.
    i : int
        The feature index to check.
    value : float
        The value to check against the feature's bounds.

    Returns
    -------
    bint
        True if the value lies within the bounds, False otherwise.
    """
    return rule.lbs[i] < value <= rule.ubs[i]



cdef int compare_rules(const Rule& a, const Rule& b):
    return a.leaf_index < b.leaf_index


cdef vector[float] get_split_point(vector[vector[Rule]] trees, int col):
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
    vector[float]
        A vector of unique split points for the specified column, excluding the smallest one.
    """

    cdef vector[float] points
    cdef vector[Rule] tree
    cdef Rule rule
    cdef vector[float].iterator it
    cdef vector[float] result

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

cdef dict count_tree_indices(cnp.ndarray[cnp.int32_t, ndim=2] array):
    cdef Py_ssize_t num_trees = array.shape[1]
    cdef Py_ssize_t num_rows = array.shape[0]
    cdef dict results = {}
    cdef dict tree_results
    cdef Py_ssize_t tree_idx, row_idx
    cdef cnp.ndarray[cnp.int32_t, ndim=1] column_values
    cdef cnp.ndarray[cnp.int32_t, ndim=1] valid_values
    
    for tree_idx in range(num_trees):
        column_values = array[:, tree_idx]
        valid_values = column_values[column_values != 0]
        tree_results = dict(Counter(valid_values))
        results[tree_idx] = tree_results
    
    return results


cdef vector[vector[Rule]] update_leaf_counts(vector[vector[Rule]] trees, object model, object back_data, str model_type):
    cdef:
        cnp.ndarray[cnp.int32_t, ndim=2] back_data_
        dict dict_counts 
        object back_data_d_matrix 

        # Variables for looping
        size_t num_trees = trees.size()
        size_t tree_size
        Rule* rule_ptr
        vector[Rule]* tree_ptr

        size_t tree_index, leaf_index
        float leaf_counts
        dict node_counts

    if model_type == "xgboost":
        back_data_d_matrix = convert_d_matrix(back_data)
        back_data_ = model.predict(back_data_d_matrix, pred_leaf=True).astype(np.int32)
        back_data_ = xgb_leaf_correction(trees, back_data_)

    elif model_type == "catboost":
        back_data_ = model.calc_leaf_indexes(back_data).astype(np.int32)

    elif model_type == "lightgbm":
        back_data_ = model.predict(back_data, pred_leaf=True).astype(np.int32)

    dict_counts = count_tree_indices(back_data_)

    for tree_index in range(num_trees):
        tree_ptr = &trees[tree_index]
        tree_size = tree_ptr[0].size()
        for leaf_index in range(tree_size):
            rule_ptr = &(tree_ptr[0][leaf_index])
            rule_ptr.count = 0.0

    for tree_index, node_counts in dict_counts.items():
        for leaf_index, leaf_counts in node_counts.items():
            tree_ptr = &trees[tree_index]
            rule_ptr = &(tree_ptr[0][leaf_index])
            rule_ptr.count = leaf_counts

    return trees