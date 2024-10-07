
from cython cimport boundscheck, wraparound, initializedcheck, nonecheck, cdivision, overflowcheck, infer_types
from libcpp.vector cimport vector
from libc.math cimport INFINITY
from libcpp.algorithm cimport sort
from libcpp.pair cimport pair
from .rule cimport create_rule, update_rule, compare_rules

ctypedef pair[double, double] RangePair

import json
import numpy as np

cdef class XGBNode:
    cdef int nodeid
    cdef str split
    cdef double split_condition
    cdef int yes
    cdef int no
    cdef list children
    cdef double leaf_value

@boundscheck(False)
@nonecheck(False)
@wraparound(False)
@initializedcheck(False)
@cdivision(True)
@overflowcheck(False)
@infer_types(False)
cdef vector[vector[Rule]] analyze_xgboost(object model, int len_col):
    cdef vector[vector[Rule]] trees = vector[vector[Rule]]()
    cdef vector[Rule] rules
    cdef dict node_dict
    cdef vector[pair[double, double]] feature_ranges
    cdef int tree_index = 0
    cdef list dumps = model.get_dump(dump_format='json')
    cdef str dump
    cdef dict tree_json

    for dump in dumps:
        tree_json = json.loads(dump)

        node_dict = {}
        parse_xgboost_node(tree_json, node_dict)


        feature_ranges = vector[pair[double, double]](len_col)
        for i in range(len_col):
            feature_ranges[i] = pair[double, double](-INFINITY, INFINITY)

        rules = vector[Rule]()

        traverse_xgboost(node_dict, 0, feature_ranges, rules, tree_index)

        sort(rules.begin(), rules.end(), compare_rules)
        trees.push_back(rules)

        tree_index += 1

    return trees


@boundscheck(False)
@nonecheck(False)
@wraparound(False)
@initializedcheck(False)
@cdivision(True)
@overflowcheck(False)
@infer_types(False)
cdef void parse_xgboost_node(dict node_json, dict node_dict):
    cdef XGBNode node = XGBNode()

    node.nodeid = node_json['nodeid']
    node.split = node_json.get('split', '')
    node.split_condition = float(node_json.get('split_condition', 0.0))
    node.yes = node_json.get('yes', -1)
    node.no = node_json.get('no', -1)
    node.children = node_json.get('children', [])
    node.leaf_value = node_json.get('leaf', np.nan)

    node_dict[node.nodeid] = node

    for child_json in node.children:
        parse_xgboost_node(child_json, node_dict)


@boundscheck(False)
@nonecheck(False)
@wraparound(False)
@initializedcheck(False)
@cdivision(True)
@overflowcheck(False)
@infer_types(False)
cdef void traverse_xgboost(dict node_dict, int nodeid, vector[pair[double, double]]& feature_ranges, vector[Rule]& rules, int tree_index):    
    cdef XGBNode node = node_dict[nodeid]
    cdef int feature_index
    cdef double threshold
    cdef pair[double, double] prev_range
    cdef Rule rule
    cdef double lb, ub

    # Check if this is a leaf node
    if not np.isnan(node.leaf_value):
        rule = create_rule(len(feature_ranges), tree_index, node.nodeid)

        # Update rule with feature ranges
        for feature_index in range(len(feature_ranges)):
            lb, ub = feature_ranges[feature_index].first, feature_ranges[feature_index].second
            if lb != -INFINITY or ub != INFINITY:
                update_rule(&rule, feature_index, lb, ub)

        rule.value = node.leaf_value
        rules.push_back(rule)
        return

    # Split node
    feature_index = int(node.split [1:])  # Extract the index after 'f'

    threshold = node.split_condition
    # Save previous range
    prev_range = feature_ranges[feature_index]

    # 'Yes' child: feature value < threshold
    feature_ranges[feature_index] = pair[double, double](feature_ranges[feature_index].first, threshold)
    traverse_xgboost(node_dict, node.yes, feature_ranges, rules, tree_index)

    # Restore previous range
    feature_ranges[feature_index] = prev_range

    # 'No' child: feature value >= threshold
    feature_ranges[feature_index] = pair[double, double](threshold, feature_ranges[feature_index].second)
    traverse_xgboost(node_dict, node.no, feature_ranges, rules, tree_index)

    # Restore previous range
    feature_ranges[feature_index] = prev_range
