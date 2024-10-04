
from libcpp.vector cimport vector
from .rule  cimport create_rule, update_rule, compare_rules
from libc.math cimport INFINITY
from libcpp.algorithm cimport sort

import pandas as pd
cimport numpy as cnp

from cython cimport boundscheck, wraparound

@boundscheck(False)
@wraparound(False)
cdef void traverse_lightgbm(object tree, str node_index, dict feature_ranges, vector[Rule]& rules, int tree_index, int len_col):
    cdef object node
    cdef int int_node_index, feature_index
    cdef Rule rule
    cdef double lb, ub, threshold
    cdef dict left_ranges, right_ranges

    # Fetch the current node
    node = tree[tree["node_index"] == node_index].iloc[0]

    # If it is a leaf node
    if pd.isna(node["split_feature"]):
        int_node_index = int(node_index[node_index.find("L") + 1:])
        rule = create_rule(len_col, tree_index, int_node_index)

        # Update rule with feature ranges
        for feature_index, (lb, ub) in feature_ranges.items():
            update_rule(&rule, feature_index, lb, ub)


        # Assign the node value to the rule
        rule.value = node["value"]
        rules.push_back(rule)
        return

    # Extract feature index and threshold for splitting
    feature_index = int(node["split_feature"].replace("Column_", ""))
    threshold = node["threshold"]

    # Copy ranges for left and right splits
    left_ranges = feature_ranges.copy()
    left_ranges[feature_index] = (
        left_ranges.get(feature_index, (-INFINITY, INFINITY))[0],
        threshold,
    )

    right_ranges = feature_ranges.copy()
    right_ranges[feature_index] = (
        threshold,
        right_ranges.get(feature_index, (-INFINITY, INFINITY))[1],
    )

    # Recursive calls for left and right child nodes
    traverse_lightgbm(tree, node["left_child"], left_ranges, rules, tree_index, len_col)
    traverse_lightgbm(tree, node["right_child"], right_ranges, rules, tree_index, len_col)


@boundscheck(False)
@wraparound(False)
cdef vector[vector[Rule]] analyze_lightgbm(object model, int len_col):
    cdef object trees_df, tree
    cdef object root_node
    cdef int tree_index
    cdef str root_index 
    
    cdef Rule rule
    cdef vector[vector[Rule]] trees
    cdef vector[Rule] temp_rules, rules = vector[Rule]()
    cdef cnp.ndarray[cnp.int64_t, ndim=1] tree_indexs

    trees_df = model.trees_to_dataframe()
    tree_indexs = trees_df.tree_index.unique()

    for tree_index in tree_indexs:
        tree = trees_df[trees_df["tree_index"] == tree_index]
        root_node = tree[tree["parent_index"].isna()].iloc[0]
        root_index = root_node["node_index"]

        traverse_lightgbm(tree, root_index, {}, rules, tree_index, len_col)

    for tree_index in tree_indexs:
        temp_rules = vector[Rule]()
        for rule in rules:
            if rule.tree_index == tree_index:
                temp_rules.push_back(rule)

        sort(temp_rules.begin(), temp_rules.end(), compare_rules)

        trees.push_back(temp_rules)

    return trees