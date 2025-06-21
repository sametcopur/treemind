from libcpp.vector cimport vector
from .rule cimport create_rule, update_rule, compare_rules, update_cats_for_rule
from libc.math cimport INFINITY
from libcpp.algorithm cimport sort

from libcpp.pair cimport pair

ctypedef pair[double, double] RangePair


cdef void traverse_lightgbm_tree(
    dict node,
    vector[RangePair]& feature_ranges,
    vector[Rule]& rules,
    int tree_index,
    vector[vector[bint]]& cat_mask  # <-- artık bool mask
):
    cdef int feature_index, i
    cdef double threshold
    cdef list threshold_split
    cdef RangePair prev_range
    cdef vector[bint] prev_mask
    cdef Rule rule
    cdef int leaf_index
    cdef size_t idx

    if "leaf_index" in node:
        leaf_index = int(node["leaf_index"])
        rule = create_rule(len(feature_ranges), tree_index, leaf_index)

        # Range koşullarını aktar
        for feature_index in range(len(feature_ranges)):
            if feature_ranges[feature_index].first != -INFINITY or feature_ranges[feature_index].second != INFINITY:
                update_rule(&rule, feature_index, feature_ranges[feature_index].first, feature_ranges[feature_index].second)

        update_cats_for_rule(&rule, cat_mask)  # artık cat_mask aktarılıyor
        rule.value = node["leaf_value"]
        rule.count = node["leaf_count"]
        rules.push_back(rule)
        return

    # Split düğümü
    feature_index = int(node["split_feature"])
    is_categorical = node["decision_type"] == "=="

    if is_categorical:
        threshold_split = node["threshold"].split("||")

        # --- mevcut maskeyi yedekle
        prev_mask = cat_mask[feature_index]        # sadece 1 kopya

        # === SOL DAL =========================================================
        # hepsini 0’a çek
        for j in range(cat_mask[feature_index].size()):
            cat_mask[feature_index][j] = 0
        # threshold setindekileri 1 yap
        for idx_str in threshold_split:
            idx = int(idx_str)
            if 0 <= idx < prev_mask.size() and prev_mask[idx]:
                cat_mask[feature_index][idx] = 1

        traverse_lightgbm_tree(node["left_child"],
                                feature_ranges, rules,
                                tree_index, cat_mask)

        # === SAĞ DAL =========================================================
        cat_mask[feature_index] = prev_mask        # yedeği geri koy
        for idx_str in threshold_split:            # eşleşenleri kapat
            idx = int(idx_str)
            if 0 <= idx < cat_mask[feature_index].size():
                cat_mask[feature_index][idx] = 0

        traverse_lightgbm_tree(node["right_child"],
                                feature_ranges, rules,
                                tree_index, cat_mask)

        # geri al
        cat_mask[feature_index] = prev_mask


    else:
        threshold = node["threshold"]
        prev_range = feature_ranges[feature_index]

        # Sol: < threshold
        feature_ranges[feature_index] = (prev_range.first, threshold)
        traverse_lightgbm_tree(node["left_child"], feature_ranges, rules, tree_index, cat_mask)

        # Sağ: >= threshold
        feature_ranges[feature_index] = (threshold, prev_range.second)
        traverse_lightgbm_tree(node["right_child"], feature_ranges, rules, tree_index, cat_mask)

        # Restore
        feature_ranges[feature_index] = prev_range


cdef tuple[vector[vector[Rule]], vector[vector[int]], vector[int]] analyze_lightgbm(object model, int len_col):
    cdef int tree_index
    cdef Rule rule
    cdef vector[vector[Rule]] trees = vector[vector[Rule]]()
    cdef vector[Rule] rules
    cdef dict node
    cdef vector[RangePair] feature_ranges

    cdef dict model_dump = model.dump_model()
    cdef list tree_list = model_dump["tree_info"]
    cdef vector[vector[int]] cat_values = vector[vector[int]]()
    cdef vector[int] cat_indices = vector[int]()
    cdef vector[vector[bint]] cat_mask
    cdef int i, cat


    cat_values.resize(len_col)
    for i, info in enumerate(model_dump["feature_infos"].values()):
        if len(info["values"]) > 0:
            cat_indices.push_back(i)

        for cat in info["values"]:
            if cat == -1:
                continue
            cat_values[i].push_back(cat)
    
    for tree_index in range(len(tree_list)):
        node = tree_list[tree_index]["tree_structure"]

        # Initialize feature_ranges
        feature_ranges = vector[RangePair](len_col)
        for i in range(len_col):
            feature_ranges[i] = (-INFINITY, INFINITY)

        # Initialize rules vector
        rules = vector[Rule]()

        cat_mask.resize(len_col)
        for i in range(len_col):
            cat_mask[i] = vector[bint](cat_values[i].size(), 1)  # boş başlat

        # Traverse the tree structure
        traverse_lightgbm_tree(node, feature_ranges, rules, tree_index, cat_mask)

        # Sort rules
        sort(rules.begin(), rules.end(), compare_rules)
        trees.push_back(rules)

    return trees, cat_values, cat_indices
