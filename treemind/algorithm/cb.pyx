from libcpp.vector cimport vector
from .rule cimport create_rule, update_rule, compare_rules, update_cats_for_rule
from libc.math cimport INFINITY
from libcpp.algorithm cimport sort
from libcpp.pair cimport pair

ctypedef pair[double, double] RangePair

import json
import tempfile
import os

cdef extract_leaf_paths_with_counts(dict model_dict):
    cdef list trees = model_dict['oblivious_trees']
    cdef dict feature_info = model_dict['features_info']
    cdef list float_features = feature_info.get('float_features', [])
    cdef int feature_count = len(float_features)

    cdef list all_trees_info = []
    cdef int tree_index, depth, num_leaves, leaf_index, num_classes
    cdef list splits, leaf_values, leaf_counts, path
    cdef dict tree, leaf_info, tree_info, class_params
    cdef str bits
    cdef int feature_idx
    cdef float border
    cdef tuple decision
    cdef double leaf_val
    cdef int leaf_cnt

    for tree_index, tree in enumerate(trees):
        splits = tree['splits']
        depth = len(splits)
        num_leaves = 1 << depth  # 2 ** depth
        leaf_values = tree['leaf_values']
        leaf_counts = tree.get('leaf_weights', [1] * num_leaves)

        class_params = model_dict['model_info'].get('class_params')
        if class_params is not None:
            num_classes = len(class_params.get('class_names', []))
            if num_classes > 2:
                raise ValueError("Multiclass CatBoost models are not supported yet.")

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
                feature_idx = split['float_feature_index']
                border = split['border']
                decision = (feature_idx, border, int(bit))  # 0 => <=, 1 => >
                path.append(decision)

            leaf_val = leaf_values[leaf_index]
            leaf_cnt = leaf_counts[leaf_index] if leaf_index < len(leaf_counts) else 1

            leaf_info = {
                'leaf_index': leaf_index,
                'path': path,
                'leaf_value': leaf_val,
                'leaf_count': leaf_cnt
            }
            tree_info['leaves'].append(leaf_info)

        all_trees_info.append(tree_info)

    return all_trees_info


cdef tuple[vector[vector[Rule]], vector[vector[int]], vector[int]] analyze_catboost(object model, int len_col):
    cdef vector[vector[Rule]] trees = vector[vector[Rule]]()
    cdef vector[Rule] rules

    # --- Kategorik yer tutucular (boş) -----------------------------------
    cdef vector[vector[int]] cat_values = vector[vector[int]]()
    cdef vector[int]        cat_indices = vector[int]()
    cat_values.resize(len_col)


    cdef dict model_json
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = os.path.join(tmp_dir, "model.json")
        model.save_model(tmp_file, format="json")
        with open(tmp_file, encoding="utf-8") as fh:
            model_json = json.load(fh)

    cdef list tree_info = extract_leaf_paths_with_counts(model_json)
    cdef int tree_index, leaf_index, feature_index
    cdef float border
    cdef int decision
    cdef Rule rule

    for tree_index in range(len(tree_info)):
        rules = vector[Rule]()
        for leaf in tree_info[tree_index]['leaves']:
            leaf_index = leaf['leaf_index']
            rule = create_rule(len_col, tree_index, leaf_index)

            for feature_index, border, decision in leaf['path']:
                if decision == 0:
                    update_rule(&rule, feature_index, -INFINITY, border)
                else:
                    update_rule(&rule, feature_index, border, INFINITY)

            # Kategorik bilgi yok ama akış gereği fonksiyona boş mask veriyoruz
            rule.cat_flags = vector[bint](len_col, 0)

            rule.value = leaf['leaf_value']
            rule.count = leaf['leaf_count']
            rules.push_back(rule)

        sort(rules.begin(), rules.end(), compare_rules)
        trees.push_back(rules)

    return trees, cat_values, cat_indices
