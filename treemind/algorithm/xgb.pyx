from libcpp.vector cimport vector
from libc.math cimport INFINITY
from libcpp.algorithm cimport sort
from libcpp.pair cimport pair
from .rule cimport create_rule, update_rule, compare_rules, update_cats_for_rule

ctypedef pair[double, double] RangePair

import json
import numpy as np
cimport numpy as cnp

import importlib

cdef class XGBNode:
    cdef int nodeid
    cdef str split
    cdef double split_condition
    cdef list categorical_condition  # kategorik koşullar için liste
    cdef int yes
    cdef int no
    cdef list children
    cdef double leaf_value
    cdef double cover
    cdef bint is_categorical  # kategorik mi değil mi


cdef tuple[vector[vector[Rule]], vector[vector[int]], vector[int], int] analyze_xgboost(object model, int len_col):
    cdef vector[vector[Rule]] trees = vector[vector[Rule]]()
    cdef vector[Rule] rules
    cdef dict node_dict
    cdef vector[RangePair] feature_ranges
    cdef int tree_index = 0
    cdef list dumps = model.get_dump(dump_format='json', with_stats=True)
    cdef str dump
    cdef dict tree_json, column_dict = (None if model.feature_names is None else dict(zip(model.feature_names, range(len(model.feature_names)))))
    cdef vector[int] cat_indices = vector[int]()
    cdef vector[vector[int]] cat_values = vector[vector[int]]()
    cdef vector[vector[bint]] cat_mask
    cdef int i
    cdef str j
    cdef bint has_cat = 0

    cat_values.resize(len_col)

    if model.feature_types is not None:
        for i, j in enumerate(model.feature_types):
            if j == "c":
                has_cat = 1
                cat_indices.push_back(i)

    if has_cat:
        cat_max = {}  # normal dict

        for dump in dumps:
            tree = json.loads(dump)
            stack = [tree]
            while stack:
                node = stack.pop()
                cond = node.get("split_condition", None)
                feat = node.get("split", None)

                if isinstance(cond, list) and feat is not None:
                    # sadece -1 olmayan kategorileri dikkate al
                    cats = [cat for cat in cond if cat != -1]
                    if cats:
                        max_cat = max(cats)
                        if feat in cat_max:
                            cat_max[feat] = max(cat_max[feat], max_cat)
                        else:
                            cat_max[feat] = max_cat

                stack.extend(node.get("children", []))

        for i, j in enumerate(model.feature_names):
            if j in cat_max.keys():
                for cat in range(cat_max[j] + 1):  # +1 eklendi
                    cat_values[i].push_back(cat)

    n_classes = len(dumps) // model.num_boosted_rounds()

    for dump in dumps:
        tree_json = json.loads(dump)

        node_dict = {}
        parse_xgboost_node(tree_json, node_dict)

        feature_ranges = vector[RangePair](len_col)
        for i in range(len_col):
            feature_ranges[i] = RangePair(-INFINITY, INFINITY)

        rules = vector[Rule]()

        # Kategorik mask'i başlat
        cat_mask.resize(len_col)
        for i in range(len_col):
            cat_mask[i] = vector[bint](cat_values[i].size(), 1)  # hepsini 1 ile başlat

        traverse_xgboost(node_dict, 0, feature_ranges, rules, tree_index, column_dict, cat_mask, cat_values)

        sort(rules.begin(), rules.end(), compare_rules)
        trees.push_back(rules)

        tree_index += 1

    return trees, cat_values, cat_indices, n_classes


cdef void parse_xgboost_node(dict node_json, dict node_dict):
    cdef XGBNode node = XGBNode()

    node.nodeid = node_json['nodeid']
    node.split = node_json.get('split', '')
    
    # Kategorik mi kontrol et
    split_condition = node_json.get('split_condition', 0.0)
    if isinstance(split_condition, list):
        node.is_categorical = True
        node.categorical_condition = split_condition
        node.split_condition = 0.0  # dummy değer
    else:
        node.is_categorical = False
        node.split_condition = float(split_condition)
        node.categorical_condition = []

    node.yes = node_json.get('yes', -1)
    node.no = node_json.get('no', -1)
    node.children = node_json.get('children', [])
    node.leaf_value = node_json.get('leaf', np.nan)
    node.cover = node_json.get('cover', np.nan)

    node_dict[node.nodeid] = node

    for child_json in node.children:
        parse_xgboost_node(child_json, node_dict)


cdef void traverse_xgboost(
    dict node_dict, 
    int nodeid, 
    vector[RangePair]& feature_ranges, 
    vector[Rule]& rules, 
    int tree_index, 
    dict column_dict,
    vector[vector[bint]]& cat_mask,
    vector[vector[int]]& cat_values
):    
    cdef XGBNode node = node_dict[nodeid]
    cdef int feature_index
    cdef double threshold
    cdef RangePair prev_range
    cdef vector[bint] prev_mask
    cdef Rule rule
    cdef double lb, ub
    cdef int cat_val
    cdef size_t idx

    # Check if this is a leaf node
    if not np.isnan(node.leaf_value):
        rule = create_rule(len(feature_ranges), tree_index, node.nodeid)

        # Update rule with feature ranges
        for feature_index in range(len(feature_ranges)):
            lb, ub = feature_ranges[feature_index].first, feature_ranges[feature_index].second
            if lb != -INFINITY or ub != INFINITY:
                update_rule(&rule, feature_index, lb, ub)

        # Kategorik maskleri güncelle
        update_cats_for_rule(&rule, cat_mask)

        rule.value = node.leaf_value
        rule.count = node.cover
        
        rules.push_back(rule)
        return

    # Split node
    if column_dict is None:
        feature_index = int(node.split[1:])  # Extract the index after 'f'
    else:
        feature_index = column_dict[node.split]

    if node.is_categorical:
        # Kategorik split işlemi
        
        # Mevcut maskeyi yedekle
        prev_mask = cat_mask[feature_index]

        # === YES BRANCH (kategorik değerler listede) ===
        # Hepsini 0'a çek
        for j in range(cat_mask[feature_index].size()):
            cat_mask[feature_index][j] = 0
        
        # Kategorik condition'daki değerleri 1 yap
        for cat_val in node.categorical_condition:
            if cat_val == -1:  # -1 genelde missing value anlamına gelir
                continue
            # cat_values içinde bu değerin indeksini bul
            for idx in range(cat_values[feature_index].size()):
                if cat_values[feature_index][idx] == cat_val and prev_mask[idx]:
                    cat_mask[feature_index][idx] = 1
                    break

        # Yes branch'i traverse et
        traverse_xgboost(node_dict, node.yes, feature_ranges, rules, tree_index, column_dict, cat_mask, cat_values)

        # === NO BRANCH (kategorik değerler listede değil) ===
        cat_mask[feature_index] = prev_mask  # Yedeği geri koy
        
        # Kategorik condition'daki değerleri 0 yap
        for cat_val in node.categorical_condition:
            if cat_val == -1:
                continue
            for idx in range(cat_values[feature_index].size()):
                if cat_values[feature_index][idx] == cat_val:
                    cat_mask[feature_index][idx] = 0
                    break

        # No branch'i traverse et
        traverse_xgboost(node_dict, node.no, feature_ranges, rules, tree_index, column_dict, cat_mask, cat_values)

        # Maskeyi geri al
        cat_mask[feature_index] = prev_mask

    else:
        # Numerik split işlemi (eski kod)
        threshold = node.split_condition
        prev_range = feature_ranges[feature_index]

        # 'Yes' child: feature value < threshold
        feature_ranges[feature_index] = RangePair(feature_ranges[feature_index].first, threshold)
        traverse_xgboost(node_dict, node.yes, feature_ranges, rules, tree_index, column_dict, cat_mask, cat_values)

        # Restore previous range
        feature_ranges[feature_index] = prev_range

        # 'No' child: feature value >= threshold
        feature_ranges[feature_index] = RangePair(threshold, feature_ranges[feature_index].second)
        traverse_xgboost(node_dict, node.no, feature_ranges, rules, tree_index, column_dict, cat_mask, cat_values)

        # Restore previous range
        feature_ranges[feature_index] = prev_range


cdef cnp.ndarray[cnp.int32_t, ndim=2, mode="c"] xgb_leaf_correction(vector[vector[Rule]] trees, int[:,:] leafs):
    cdef:
        list leaf_indices, mappings = [], max_indices = []
        vector[Rule] tree
        int max_index, i
        Rule rule
        cnp.ndarray[cnp.int32_t, ndim=2, mode="c"] leafs_arr = np.asarray(leafs)
        cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] key, values, mapping_array
        cnp.ndarray[cnp.npy_bool, ndim=1, mode="c"] mask
        
    for tree in trees:
        leaf_indices = [rule.leaf_index for rule in tree]
        max_index = max(leaf_indices)
        mapping_array = np.full(max_index + 1, -1, dtype=np.int32)
        
        for i, rule in enumerate(tree):
            mapping_array[rule.leaf_index] = i
        
        mappings.append(mapping_array)
        max_indices.append(max_index)

    for i, (mapping_array, max_index) in enumerate(zip(mappings, max_indices)):
        mask = leafs_arr[:, i] <= max_index
        leafs_arr[mask, i] = mapping_array[leafs_arr[mask, i]]

    return leafs_arr


cdef object convert_d_matrix(object x):
    DMatrix = getattr(importlib.import_module("xgboost"), "DMatrix")
    return DMatrix(x)