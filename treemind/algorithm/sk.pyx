from .rule cimport create_rule, update_rule, compare_rules, update_cats_for_rule
from libc.math cimport INFINITY, log, exp
from libcpp.algorithm cimport sort
from libcpp.pair cimport pair
from libcpp.vector cimport vector

ctypedef pair[double, double] RangePair


cdef double convert_sklearn_to_logit(double prob):
    """Convert sklearn probability/proportion to logit space like LightGBM"""
    cdef double epsilon = 1e-15
    cdef double max_logit = 700.0
    
    if prob <= epsilon:
        prob = epsilon
    elif prob >= (1.0 - epsilon):
        prob = 1.0 - epsilon
    
    cdef double logit = log(prob / (1.0 - prob))
    
    if logit > max_logit:
        return max_logit
    elif logit < -max_logit:
        return -max_logit
    else:
        return logit


cdef vector[bint] bitset_to_categorical_mask(object bitset_row, int max_categories):
    """Bitset'i kategorik mask'e çevir"""
    cdef vector[bint] mask = vector[bint](max_categories, 0)
    cdef int word_idx, bit_idx, cat_id
    cdef unsigned int bitset_value
    
    for word_idx in range(len(bitset_row)):
        bitset_value = bitset_row[word_idx]
        if bitset_value == 0:
            continue
            
        for bit_idx in range(32):
            if bitset_value & (1 << bit_idx):
                cat_id = word_idx * 32 + bit_idx
                if cat_id < max_categories:
                    mask[cat_id] = 1
    
    return mask


cdef vector[bint] get_right_categories_mask(vector[bint]& left_mask, vector[bint]& all_categories_mask):
    """Sol kategorilerden sağ kategorileri çıkar"""
    cdef vector[bint] right_mask = vector[bint](all_categories_mask.size(), 0)
    cdef int i
    
    for i in range(all_categories_mask.size()):
        if all_categories_mask[i] == 1 and left_mask[i] == 0:
            right_mask[i] = 1
    
    return right_mask


cdef vector[int] create_bin_to_original_mapping(object model, int len_col):
    """Bin mapper feature index'ini orijinal feature index'ine çeviren mapping oluştur"""
    cdef vector[int] bin_to_original = vector[int]()
    cdef int i
    
    if not hasattr(model, 'is_categorical_') or not hasattr(model, '_bin_mapper') or model.is_categorical_ is None:
        # Mapping yok - 1:1 mapping varsay
        for i in range(len_col):
            bin_to_original.push_back(i)
        return bin_to_original
    
    # Orijinal feature'lardaki kategorik ve numerik indeksleri bul
    cdef list original_categorical_indices = []
    cdef list original_numeric_indices = []
    
    for i in range(len(model.is_categorical_)):
        if model.is_categorical_[i]:
            original_categorical_indices.append(i)
        else:
            original_numeric_indices.append(i)
    
    # Bin mapper sırası: önce kategorik, sonra numerik
    for cat_idx in original_categorical_indices:
        bin_to_original.push_back(cat_idx)
    
    for num_idx in original_numeric_indices:
        bin_to_original.push_back(num_idx)
    
    return bin_to_original


cdef void traverse_sklearn_tree(
    object tree,
    int node_id,
    vector[RangePair]& feature_ranges,
    vector[Rule]& rules,
    int tree_index,
    double scale_factor,
    bint is_classification,
    int n_classes,
    int class_idx
):
    cdef int feature_index, left_child, right_child, i
    cdef double threshold, raw_value, prob, logit_value
    cdef RangePair prev_range
    cdef Rule rule

    # Leaf node check
    if tree.children_left[node_id] == tree.children_right[node_id]:
        rule = create_rule(len(feature_ranges), tree_index, node_id)
        
        # Add range constraints
        for i in range(len(feature_ranges)):
            if feature_ranges[i].first != -INFINITY or feature_ranges[i].second != INFINITY:
                update_rule(&rule, i, feature_ranges[i].first, feature_ranges[i].second)

        rule.cat_flags = vector[bint](len(feature_ranges), 0)
        
        # Process leaf value
        if is_classification:
            if n_classes > 2:
                if tree.value[node_id].shape[1] > class_idx:
                    class_count = tree.value[node_id][0][class_idx]
                    total_samples = tree.n_node_samples[node_id]
                    prob = class_count / total_samples if total_samples > 0 else (1.0 / n_classes)
                    logit_value = convert_sklearn_to_logit(prob)
                    rule.value = logit_value * scale_factor
                else:
                    rule.value = 0.0
            else:
                if tree.value[node_id].shape[1] >= 2:
                    positive_count = tree.value[node_id][0][1]
                    total_samples = tree.n_node_samples[node_id]
                    prob = positive_count / total_samples if total_samples > 0 else 0.5
                    logit_value = convert_sklearn_to_logit(prob)
                    rule.value = logit_value * scale_factor
                else:
                    raw_value = tree.value[node_id][0][0]
                    rule.value = raw_value * scale_factor
        else:
            raw_value = tree.value[node_id][0][0]
            rule.value = raw_value * scale_factor
            
        rule.count = tree.n_node_samples[node_id]
        rules.push_back(rule)
        return

    # Internal node
    feature_index = tree.feature[node_id]
    threshold = tree.threshold[node_id]
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]
    prev_range = feature_ranges[feature_index]
    
    # Left child (feature <= threshold)
    feature_ranges[feature_index] = (prev_range.first, threshold)
    traverse_sklearn_tree(tree, left_child, feature_ranges, rules, tree_index, scale_factor, is_classification, n_classes, class_idx)
    
    # Right child (feature > threshold)
    feature_ranges[feature_index] = (threshold, prev_range.second)
    traverse_sklearn_tree(tree, right_child, feature_ranges, rules, tree_index, scale_factor, is_classification, n_classes, class_idx)
    
    # Restore original range
    feature_ranges[feature_index] = prev_range


cdef bint has_valid_categories(vector[bint]& mask):
    """Kategorik mask'in geçerli kategorileri olup olmadığını kontrol et"""
    cdef int i
    for i in range(mask.size()):
        if mask[i] == 1:
            return True
    return False


cdef void traverse_histgb_tree(
    object nodes,
    int node_id,
    vector[RangePair]& feature_ranges,
    vector[Rule]& rules,
    int tree_index,
    double scale_factor,
    vector[vector[bint]]& cat_mask,
    vector[bint]& is_categorical_features,
    vector[vector[bint]]& all_categories_mask,
    object predictor,
    vector[int]& bin_to_original_mapping
):
    """HistGradientBoosting tree traversal"""
    cdef int bin_feature_idx, original_feature_idx, left_child, right_child, i, bitset_idx
    cdef double threshold, raw_value
    cdef RangePair prev_range
    cdef vector[bint] prev_mask, left_mask, right_mask
    cdef Rule rule
    cdef bint is_categorical_split
    
    # Leaf node
    if nodes[node_id]['is_leaf'] == 1:
        rule = create_rule(len(feature_ranges), tree_index, node_id)
        
        # Add numeric constraints
        for i in range(len(feature_ranges)):
            if feature_ranges[i].first != -INFINITY or feature_ranges[i].second != INFINITY:
                update_rule(&rule, i, feature_ranges[i].first, feature_ranges[i].second)
        
        # Add categorical constraints
        update_cats_for_rule(&rule, cat_mask)
        
        raw_value = nodes[node_id]['value']
        rule.value = raw_value * scale_factor
        rule.count = nodes[node_id]['count']
        
        rules.push_back(rule)
        return
    
    # Internal node
    bin_feature_idx = nodes[node_id]['feature_idx']
    original_feature_idx = bin_to_original_mapping[bin_feature_idx] if bin_feature_idx < bin_to_original_mapping.size() else bin_feature_idx
    
    left_child = nodes[node_id]['left']
    right_child = nodes[node_id]['right']
    
    is_categorical_split = (original_feature_idx < is_categorical_features.size() and 
                           is_categorical_features[original_feature_idx] == 1 and 
                           nodes[node_id]['is_categorical'] == 1)
    
    if is_categorical_split:
        # Categorical split
        bitset_idx = nodes[node_id]['bitset_idx']
        
        if (hasattr(predictor, 'raw_left_cat_bitsets') and 
            predictor.raw_left_cat_bitsets is not None and 
            bitset_idx < len(predictor.raw_left_cat_bitsets)):
            
            prev_mask = cat_mask[original_feature_idx]
            
            # Get left categories from bitset
            left_mask = bitset_to_categorical_mask(
                predictor.raw_left_cat_bitsets[bitset_idx], 
                all_categories_mask[original_feature_idx].size()
            )
            
            # Left child: left_mask categories
            if has_valid_categories(left_mask):
                cat_mask[original_feature_idx] = left_mask
                traverse_histgb_tree(nodes, left_child, feature_ranges, rules, tree_index,
                                   scale_factor, cat_mask, is_categorical_features, 
                                   all_categories_mask, predictor, bin_to_original_mapping)
            
            # Right child: remaining categories
            right_mask = get_right_categories_mask(left_mask, all_categories_mask[original_feature_idx])
            if has_valid_categories(right_mask):
                cat_mask[original_feature_idx] = right_mask
                traverse_histgb_tree(nodes, right_child, feature_ranges, rules, tree_index,
                                   scale_factor, cat_mask, is_categorical_features, 
                                   all_categories_mask, predictor, bin_to_original_mapping)
            
            # Restore original mask
            cat_mask[original_feature_idx] = prev_mask
        else:
            # Fallback to numeric split
            threshold = float(nodes[node_id]['num_threshold'])
            prev_range = feature_ranges[original_feature_idx]
            
            # Left: <= threshold
            feature_ranges[original_feature_idx] = (prev_range.first, threshold)
            traverse_histgb_tree(nodes, left_child, feature_ranges, rules, tree_index,
                               scale_factor, cat_mask, is_categorical_features, 
                               all_categories_mask, predictor, bin_to_original_mapping)
            
            # Right: > threshold
            feature_ranges[original_feature_idx] = (threshold, prev_range.second)
            traverse_histgb_tree(nodes, right_child, feature_ranges, rules, tree_index,
                               scale_factor, cat_mask, is_categorical_features, 
                               all_categories_mask, predictor, bin_to_original_mapping)
            
            feature_ranges[original_feature_idx] = prev_range
    else:
        # Numeric split
        threshold = float(nodes[node_id]['num_threshold'])
        prev_range = feature_ranges[original_feature_idx]
        
        # Left: <= threshold
        feature_ranges[original_feature_idx] = (prev_range.first, threshold)
        traverse_histgb_tree(nodes, left_child, feature_ranges, rules, tree_index,
                           scale_factor, cat_mask, is_categorical_features, 
                           all_categories_mask, predictor, bin_to_original_mapping)
        
        # Right: > threshold
        feature_ranges[original_feature_idx] = (threshold, prev_range.second)
        traverse_histgb_tree(nodes, right_child, feature_ranges, rules, tree_index,
                           scale_factor, cat_mask, is_categorical_features, 
                           all_categories_mask, predictor, bin_to_original_mapping)
        
        feature_ranges[original_feature_idx] = prev_range


cdef void initialize_categorical_info(
    object model,
    int len_col,
    vector[bint]& is_categorical_features,
    vector[vector[bint]]& all_categories_mask,
    vector[vector[int]]& cat_values,
    vector[int]& cat_indices
):
    """Kategorik bilgileri initialize et"""
    cdef int i, bitset_idx, cat_val, original_feature_idx
    cdef list original_categorical_indices
    
    is_categorical_features = vector[bint](len_col, 0)
    all_categories_mask = vector[vector[bint]](len_col)
    cat_values = vector[vector[int]](len_col)
    
    if not (hasattr(model, 'is_categorical_') and model.is_categorical_ is not None):
        return
    
    # Kategorik feature'ları belirle
    original_categorical_indices = []
    for i in range(min(len(model.is_categorical_), len_col)):
        if model.is_categorical_[i]:
            is_categorical_features[i] = 1
            cat_indices.push_back(i)
            original_categorical_indices.append(i)
    
    # Bitset bilgilerini al
    if not (hasattr(model, '_bin_mapper') and 
            hasattr(model._bin_mapper, 'make_known_categories_bitsets') and 
            callable(getattr(model._bin_mapper, 'make_known_categories_bitsets'))):
        return
    
    known_categories_bitsets, _ = model._bin_mapper.make_known_categories_bitsets()
    
    for bitset_idx in range(min(len(known_categories_bitsets), len(original_categorical_indices))):
        original_feature_idx = original_categorical_indices[bitset_idx]
        
        if original_feature_idx < len_col:
            bitset_data = known_categories_bitsets[bitset_idx]
            max_possible_categories = len(bitset_data) * 32
            
            all_categories_mask[original_feature_idx] = bitset_to_categorical_mask(
                bitset_data, max_possible_categories
            )
            
            # cat_values güncelle
            for cat_val in range(all_categories_mask[original_feature_idx].size()):
                if all_categories_mask[original_feature_idx][cat_val] == 1:
                    cat_values[original_feature_idx].push_back(cat_val)

cdef void process_histgb_predictor(
    object predictor,
    int tree_index,
    int class_idx,
    int n_classes,
    double scale_factor,
    int len_col,
    vector[bint]& is_categorical_features,
    vector[vector[bint]]& all_categories_mask,
    vector[int]& bin_to_original_mapping,
    vector[vector[Rule]]& trees
):
    """HistGB predictor'ı işle"""
    cdef vector[RangePair] feature_ranges = vector[RangePair](len_col)
    cdef vector[vector[bint]] cat_mask
    cdef vector[Rule] rules
    cdef int i, effective_tree_index
    
    # Initialize feature ranges
    for i in range(len_col):
        feature_ranges[i] = (-INFINITY, INFINITY)
    
    # Initialize categorical mask
    cat_mask.resize(len_col)
    for i in range(len_col):
        cat_mask[i] = all_categories_mask[i]
    
    effective_tree_index = tree_index * n_classes + class_idx if n_classes > 1 else tree_index
    
    traverse_histgb_tree(predictor.nodes, 0, feature_ranges, rules, effective_tree_index,
                        scale_factor, cat_mask, is_categorical_features, 
                        all_categories_mask, predictor, bin_to_original_mapping)
    
    sort(rules.begin(), rules.end(), compare_rules)
    trees.push_back(rules)


cdef tuple[vector[vector[Rule]], vector[vector[int]], vector[int]] analyze_sklearn(object model, int len_col, int n_classes):
    cdef vector[vector[Rule]] trees = vector[vector[Rule]]()
    cdef vector[vector[int]] cat_values = vector[vector[int]]()
    cdef vector[int] cat_indices = vector[int]()
    cdef double scale_factor = 1.0
    cdef bint is_classification = False
    cdef bint is_histgb = False
    cdef int tree_index, class_idx, i
    
    # Categorical support variables
    cdef vector[bint] is_categorical_features
    cdef vector[vector[bint]] all_categories_mask
    cdef vector[int] bin_to_original_mapping
    
    # Determine model type
    model_name = model.__class__.__name__
    is_classification = 'Classifier' in model_name
    is_histgb = 'HistGradientBoosting' in model_name
    
    # Initialize categorical information
    if is_histgb:
        initialize_categorical_info(model, len_col, is_categorical_features, 
                                  all_categories_mask, cat_values, cat_indices)
        bin_to_original_mapping = create_bin_to_original_mapping(model, len_col)
    else:
        is_categorical_features = vector[bint](len_col, 0)
        all_categories_mask = vector[vector[bint]](len_col)
        cat_values = vector[vector[int]](len_col)
    
    # Process trees based on model type
    if is_histgb:
        predictors = model._predictors

        scale_factor = getattr(model, 'learning_rate', 1.0)
        
        for tree_index in range(len(predictors)):
            predictor_group = predictors[tree_index]
            
            if isinstance(predictor_group, list):
                # Multiple classes
                for class_idx in range(min(len(predictor_group), n_classes)):
                    process_histgb_predictor(predictor_group[class_idx], tree_index, class_idx,
                                           n_classes, scale_factor, len_col, is_categorical_features,
                                           all_categories_mask, bin_to_original_mapping, trees)
            else:
                # Single class/regression
                process_histgb_predictor(predictor_group, tree_index, 0, 1, scale_factor, len_col,
                                       is_categorical_features, all_categories_mask, 
                                       bin_to_original_mapping, trees)
    
    elif hasattr(model, 'estimators_'):
        # Standard ensemble methods
        trees_list = []
        if 'RandomForest' in model_name or 'ExtraTrees' in model_name:
            trees_list = [est.tree_ for est in model.estimators_]
            scale_factor = 1.0 / len(trees_list)
        elif 'GradientBoosting' in model_name:
            for i in range(len(model.estimators_)):
                trees_list.append(model.estimators_[i, 0].tree_)
            scale_factor = model.learning_rate
        elif 'AdaBoost' in model_name:
            trees_list = [est.tree_ for est in model.estimators_]
            scale_factor = 1.0
        else:
            trees_list = [est.tree_ for est in model.estimators_]
            scale_factor = 1.0 / len(trees_list)
        
        # Process each tree
        for tree_index, tree in enumerate(trees_list):
            if is_classification and n_classes > 2:
                # Multiclass
                for class_idx in range(n_classes):
                    feature_ranges = vector[RangePair](len_col)
                    for i in range(len_col):
                        feature_ranges[i] = (-INFINITY, INFINITY)
                    
                    rules = vector[Rule]()
                    effective_tree_index = tree_index * n_classes + class_idx
                    
                    traverse_sklearn_tree(tree, 0, feature_ranges, rules, effective_tree_index, 
                                        scale_factor, is_classification, n_classes, class_idx)
                    
                    sort(rules.begin(), rules.end(), compare_rules)
                    trees.push_back(rules)
            else:
                # Binary classification or regression
                feature_ranges = vector[RangePair](len_col)
                for i in range(len_col):
                    feature_ranges[i] = (-INFINITY, INFINITY)
                
                rules = vector[Rule]()
                
                traverse_sklearn_tree(tree, 0, feature_ranges, rules, tree_index, 
                                    scale_factor, is_classification, n_classes, 0)
                
                sort(rules.begin(), rules.end(), compare_rules)
                trees.push_back(rules)
    
    else:
        tree = model.tree_
        feature_ranges = vector[RangePair](len_col)
        for i in range(len_col):
            feature_ranges[i] = (-INFINITY, INFINITY)
        
        rules = vector[Rule]()
        
        traverse_sklearn_tree(tree, 0, feature_ranges, rules, 0, 1.0, is_classification, n_classes, 0)
        
        sort(rules.begin(), rules.end(), compare_rules)
        trees.push_back(rules)
    
    return trees, cat_values, cat_indices