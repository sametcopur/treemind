from libcpp.vector cimport vector
from .rule cimport create_rule, update_rule, compare_rules
from libc.math cimport INFINITY
from libcpp.algorithm cimport sort

import pandas as pd
cimport numpy as cnp

from libcpp.pair cimport pair

ctypedef pair[double, double] RangePair

from cython cimport boundscheck, wraparound, initializedcheck, nonecheck, cdivision, overflowcheck, infer_types
@boundscheck(False)
@nonecheck(False)
@wraparound(False)
@initializedcheck(False)
@cdivision(True)
@overflowcheck(False)
@infer_types(False)
cdef void traverse_lightgbm_tree(dict node, vector[RangePair]& feature_ranges, vector[Rule]& rules, int tree_index):
    cdef int int_node_index, feature_index
    cdef Rule rule
    cdef double lb, ub, threshold
    cdef RangePair prev_range

    if "leaf_index" in node:
        # Leaf node
        int_node_index = int(node["leaf_index"])
        rule = create_rule(len(feature_ranges), tree_index, int_node_index)

        # Update rule with feature ranges
        for feature_index in range(len(feature_ranges)):
            lb, ub = feature_ranges[feature_index]
            if lb != -INFINITY or ub != INFINITY:
                update_rule(&rule, feature_index, lb, ub)

        rule.value = node["leaf_value"]
        rule.count = node["leaf_count"]
        rules.push_back(rule)
        return

    # Split node
    feature_index = int(node["split_feature"])
    threshold = node["threshold"]

    # Save previous range
    prev_range = feature_ranges[feature_index]

    # Update feature range for left child
    feature_ranges[feature_index] = (feature_ranges[feature_index].first, threshold)
    traverse_lightgbm_tree(node["left_child"], feature_ranges, rules, tree_index)

    # Restore previous range
    feature_ranges[feature_index] = prev_range

    # Update feature range for right child
    feature_ranges[feature_index] = (threshold, feature_ranges[feature_index].second)
    traverse_lightgbm_tree(node["right_child"], feature_ranges, rules, tree_index)

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
    cdef int tree_index
    cdef Rule rule
    cdef vector[vector[Rule]] trees = vector[vector[Rule]]()
    cdef vector[Rule] rules
    cdef dict node
    cdef vector[RangePair] feature_ranges


    cdef list tree_list = model.dump_model()["tree_info"]

    for tree_index in range(len(tree_list)):
        node = tree_list[tree_index]["tree_structure"]

        # Initialize feature_ranges
        feature_ranges = vector[RangePair](len_col)
        for i in range(len_col):
            feature_ranges[i] = (-INFINITY, INFINITY)

        # Initialize rules vector
        rules = vector[Rule]()

        # Traverse the tree structure
        traverse_lightgbm_tree(node, feature_ranges, rules, tree_index)

        # Sort rules
        sort(rules.begin(), rules.end(), compare_rules)
        trees.push_back(rules)

    return trees
