from libcpp.vector cimport vector
from .rule cimport create_rule, update_rule, compare_rules
from libc.math cimport INFINITY
from libcpp.algorithm cimport sort

from libcpp.pair cimport pair

ctypedef pair[double, double] RangePair

import json
import tempfile
import os


cdef extract_leaf_paths_with_counts(dict model_dict):
    cdef list trees 
    cdef dict feature_info
    cdef list float_features 
    cdef int feature_count 
    cdef list splits 
    cdef int depth
    cdef int num_leaves
    cdef list leaf_values
    cdef list leaf_counts 
    cdef int feature_index
    cdef float border
    cdef dict class_params 
    cdef int num_classes 
    cdef list leaf_values_per_leaf
    cdef dict tree_info 
    cdef int leaf_index
    cdef str bits
    cdef list path
    cdef tuple decision
    cdef double leaf_value
    cdef int leaf_count
    cdef dict leaf_info
    cdef list all_trees_info = []
    cdef int tree_index
    cdef dict tree

    trees = model_dict['oblivious_trees']
    feature_info = model_dict['features_info']
    float_features = feature_info.get('float_features', [])
    feature_count = len(float_features)

    for tree_index, tree in enumerate(trees):
        splits = tree['splits']
        depth = len(splits)
        num_leaves = int(2 ** depth)
        leaf_values = tree['leaf_values']
        leaf_counts = tree.get('leaf_weights', [1] * num_leaves) 

        class_params = model_dict['model_info'].get('class_params')

        # Determine the number of classes
        if class_params is not None:
            num_classes = len(class_params.get('class_names', []))

            # If class_params exist and indicate multi-class, raise an error
            if num_classes > 2:
                raise ValueError("Multiclass catboost models are not supported yet.")

        # At this point, we are assured that it's a regression model
        # Adjust leaf_values for regression
        leaf_values_per_leaf = leaf_values

        tree_info = {
            'tree_index': tree_index,
            'feature_count': feature_count,
            'depth': depth,
            'num_leaves': num_leaves,
            'leaves': []
        }


        for leaf_index in range(num_leaves):
            bits = bin(leaf_index)[2:].zfill(depth)
            path = []
            for bit, split in zip(bits, splits):
                feature_index = split['float_feature_index']
                border = split['border']
                decision = (feature_index, border, int(bit))  # bit == 0 for <=, else >
                path.append(decision)

            # Collect the leaf value(s) and count for each leaf
            leaf_value = leaf_values_per_leaf[leaf_index]
            leaf_count = leaf_counts[leaf_index] if leaf_index < len(leaf_counts) else 1

            leaf_info = {
                'leaf_index': leaf_index,
                'path': path,
                'leaf_value': leaf_value,
                'leaf_count': leaf_count
            }
            tree_info['leaves'].append(leaf_info)
        all_trees_info.append(tree_info)
    
    return all_trees_info



cdef vector[vector[Rule]] analyze_catboost(object model, int len_col):
    cdef int tree_index
    cdef Rule rule
    cdef vector[vector[Rule]] trees = vector[vector[Rule]]()
    cdef vector[Rule] rules
    cdef dict node
    cdef vector[RangePair] feature_ranges
    cdef dict model_json
    cdef list tree_list

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = os.path.join(tmp_dir, "model.json")
        model.save_model(tmp_file, format="json")
        with open(tmp_file, encoding="utf-8") as fh:
            model_json = json.load(fh)

    cdef list tree_info = extract_leaf_paths_with_counts(model_json)

    for tree_index, tree_data in enumerate(tree_info):
        rules = vector[Rule]()

        for leaf_info in tree_data['leaves']:
            rule = create_rule(len_col, tree_index, leaf_info['leaf_index'])

            for feature_index, border, decision in leaf_info['path']:
                if decision == 0:
                    update_rule(&rule, feature_index, -INFINITY, border)
                else:
                    update_rule(&rule, feature_index, border, INFINITY)

            rule.value = leaf_info['leaf_value']
            rule.count = leaf_info['leaf_count']
            rules.push_back(rule)

        # Sort rules
        sort(rules.begin(), rules.end(), compare_rules)
        trees.push_back(rules)

    return trees