from libcpp.vector cimport vector
from .rule cimport create_rule, update_rule, compare_rules
from libc.math cimport INFINITY
from libcpp.algorithm cimport sort

import pandas as pd
cimport numpy as cnp

from cython cimport boundscheck, wraparound, initializedcheck, nonecheck, cdivision, overflowcheck, infer_types


cdef class Node:
    cdef str node_index
    cdef str left_child
    cdef str right_child
    cdef int split_feature  # -1 if leaf node
    cdef double threshold
    cdef double value
    cdef str parent_index

@boundscheck(False)
@nonecheck(False)
@wraparound(False)
@initializedcheck(False)
@cdivision(True)
@overflowcheck(False)
@infer_types(False)
cdef void traverse_lightgbm(dict node_dict, str node_index, vector[RangePair]& feature_ranges, vector[Rule]& rules, int tree_index):
    cdef Node node = node_dict[node_index]
    cdef int int_node_index, feature_index
    cdef Rule rule
    cdef double lb, ub, threshold
    cdef RangePair prev_range

    if node.split_feature == -1:
        # Leaf node
        int_node_index = int(node_index[node_index.find("L") + 1:])
        rule = create_rule(len(feature_ranges), tree_index, int_node_index)

        # Update rule with feature ranges
        for feature_index in range(len(feature_ranges)):
            lb, ub = feature_ranges[feature_index]
            if lb != -INFINITY or ub != INFINITY:
                update_rule(&rule, feature_index, lb, ub)

        rule.value = node.value
        rules.push_back(rule)
        return

    # Split node
    feature_index = node.split_feature
    threshold = node.threshold

    # Save previous range
    prev_range = feature_ranges[feature_index]

    # Update feature range for left child
    feature_ranges[feature_index] = (feature_ranges[feature_index].first, threshold)
    traverse_lightgbm(node_dict, node.left_child, feature_ranges, rules, tree_index)

    # Restore previous range
    feature_ranges[feature_index] = prev_range

    # Update feature range for right child
    feature_ranges[feature_index] = (threshold, feature_ranges[feature_index].second)
    traverse_lightgbm(node_dict, node.right_child, feature_ranges, rules, tree_index)

    # Restore previous range
    feature_ranges[feature_index] = prev_range

@boundscheck(False)
@nonecheck(False)
@wraparound(False)
@initializedcheck(False)
@cdivision(True)
@overflowcheck(False)
@infer_types(False)
cdef vector[vector[Rule]] analyze_lightgbm(object model, int len_col):
    cdef object trees_df, tree_df
    cdef int tree_index
    cdef str root_index
    cdef Rule rule
    cdef vector[vector[Rule]] trees = vector[vector[Rule]]()
    cdef vector[Rule] rules
    cdef cnp.ndarray[cnp.int64_t, ndim=1] tree_indices
    cdef dict node_dict
    cdef Node node, root_node
    cdef vector[RangePair] feature_ranges

    trees_df = model.trees_to_dataframe()
    tree_indices = trees_df.tree_index.unique()

    for tree_index in tree_indices:
        tree_df = trees_df[trees_df["tree_index"] == tree_index]

        # Build node_dict
        node_dict = {}
        for idx in tree_df.index:
            row = tree_df.loc[idx]
            node = Node()
            node.node_index = row["node_index"]
            node.left_child = row["left_child"]
            node.right_child = row["right_child"]
            node.parent_index = row["parent_index"]

            split_feature = row["split_feature"]
            if pd.isna(split_feature):
                node.split_feature = -1  # Leaf node
            else:
                node.split_feature = int(split_feature.replace("Column_", ""))
            node.threshold = row["threshold"]
            node.value = row["value"]
            node_dict[node.node_index] = node

        # Find root node
        for node in node_dict.values():
            if pd.isna(node.parent_index):
                root_node = node
                break

        # Initialize feature_ranges
        feature_ranges = vector[RangePair](len_col)
        for i in range(len_col):
            feature_ranges[i] = (-INFINITY, INFINITY)

        # Initialize rules vector
        rules = vector[Rule]()

        # Traverse tree
        traverse_lightgbm(node_dict, root_node.node_index, feature_ranges, rules, tree_index)

        # Sort rules
        sort(rules.begin(), rules.end(), compare_rules)
        trees.push_back(rules)

    return trees
