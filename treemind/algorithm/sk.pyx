from .rule cimport create_rule, update_rule, compare_rules, update_cats_for_rule
from libc.math cimport INFINITY, log, exp
from libcpp.algorithm cimport sort
from libcpp.pair cimport pair
from libcpp.vector cimport vector

ctypedef pair[double, double] RangePair


cdef double convert_sklearn_to_logit(double prob):
    """Convert sklearn probability/proportion to logit space like LightGBM"""
    cdef double epsilon = 1e-15  # Small value to prevent log(0)
    cdef double max_logit = 700.0  # Prevent overflow
    
    # Clamp probability to safe range
    if prob <= epsilon:
        prob = epsilon
    elif prob >= (1.0 - epsilon):
        prob = 1.0 - epsilon
    
    # Convert to logit: log(p / (1 - p))
    cdef double logit = log(prob / (1.0 - prob))
    
    # Clamp to prevent overflow
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
    cdef int original_idx, bin_idx
    
    if not hasattr(model, 'is_categorical_') or not hasattr(model, '_bin_mapper'):
        # Mapping yok - 1:1 mapping varsay
        for i in range(len_col):
            bin_to_original.push_back(i)
        return bin_to_original

    elif model.is_categorical_ is None:
        for i in range(len_col):
            bin_to_original.push_back(i)
        return bin_to_original
    
    # Orijinal feature'lardaki kategorik olanların indekslerini bul
    original_categorical_indices = []
    for original_idx in range(len(model.is_categorical_)):
        if model.is_categorical_[original_idx]:
            original_categorical_indices.append(original_idx)
    
    # Orijinal feature'lardaki numerik olanların indekslerini bul
    original_numeric_indices = []
    for original_idx in range(len(model.is_categorical_)):
        if not model.is_categorical_[original_idx]:
            original_numeric_indices.append(original_idx)
    
    # Bin mapper sırası: önce kategorik, sonra numerik
    # Kategorik feature'ları ekle
    for cat_idx in original_categorical_indices:
        bin_to_original.push_back(cat_idx)
    
    # Numerik feature'ları ekle
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
    cdef int feature_index
    cdef double threshold
    cdef RangePair prev_range
    cdef Rule rule
    cdef int left_child, right_child
    cdef double raw_value, prob, logit_value
    cdef int i

    # Standard sklearn leaf check
    if tree.children_left[node_id] == tree.children_right[node_id]:
        # Leaf node - create rule
        rule = create_rule(len(feature_ranges), tree_index, node_id)
        
        # Add range constraints
        for i in range(len(feature_ranges)):
            if feature_ranges[i].first != -INFINITY or feature_ranges[i].second != INFINITY:
                update_rule(&rule, i, feature_ranges[i].first, feature_ranges[i].second)

        # Set categorical flags (all False for sklearn)
        rule.cat_flags = vector[bint](len(feature_ranges), 0)
        
        # Process leaf value
        if is_classification:
            if n_classes > 2:
                # Multiclass
                if tree.value[node_id].shape[1] > class_idx:
                    class_count = tree.value[node_id][0][class_idx]
                    total_samples = tree.n_node_samples[node_id]
                    prob = class_count / total_samples if total_samples > 0 else (1.0 / n_classes)
                    
                    logit_value = convert_sklearn_to_logit(prob)
                    rule.value = logit_value * scale_factor
                else:
                    rule.value = 0.0
            else:
                # Binary classification
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
            # Regression
            raw_value = tree.value[node_id][0][0]
            rule.value = raw_value * scale_factor
            
        rule.count = tree.n_node_samples[node_id]
        rules.push_back(rule)
        return

    # Internal node - split
    feature_index = tree.feature[node_id]
    threshold = tree.threshold[node_id]
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]
    
    # Store current range for this feature
    prev_range = feature_ranges[feature_index]
    
    # Traverse left child (feature <= threshold)
    feature_ranges[feature_index] = (prev_range.first, threshold)
    traverse_sklearn_tree(tree, left_child, feature_ranges, rules, tree_index, scale_factor, is_classification, n_classes, class_idx)
    
    # Traverse right child (feature > threshold)
    feature_ranges[feature_index] = (threshold, prev_range.second)
    traverse_sklearn_tree(tree, right_child, feature_ranges, rules, tree_index, scale_factor, is_classification, n_classes, class_idx)
    
    # Restore original range
    feature_ranges[feature_index] = prev_range


cdef int map_bin_to_original_feature(int bin_feature_idx, vector[int]& bin_to_original_mapping):
    """Bin mapper feature index'ini orijinal feature index'ine çevir"""
    if bin_feature_idx < 0 or bin_feature_idx >= bin_to_original_mapping.size():
        return bin_feature_idx  # Güvenli fallback
    
    return bin_to_original_mapping[bin_feature_idx]


cdef bint is_original_feature_categorical(int original_feature_idx, vector[bint]& is_categorical_features):
    """Orijinal feature index'in kategorik olup olmadığını kontrol et"""
    if original_feature_idx < 0 or original_feature_idx >= is_categorical_features.size():
        return False
    
    return is_categorical_features[original_feature_idx]


cdef void traverse_histgb_tree(
    object nodes,
    int node_id,
    vector[RangePair]& feature_ranges,
    vector[Rule]& rules,
    int tree_index,
    double scale_factor,
    bint is_classification,
    int n_classes,
    int class_idx,
    vector[vector[bint]]& cat_mask,
    vector[bint]& is_categorical_features,
    vector[vector[bint]]& all_categories_mask,
    object predictor,
    vector[int]& bin_to_original_mapping
):
    """HistGradientBoosting tree traversal - feature mapping ile düzeltilmiş"""
    cdef int bin_feature_idx, original_feature_idx, left_child, right_child, i
    cdef double threshold, raw_value
    cdef RangePair prev_range
    cdef vector[bint] prev_mask, left_mask, right_mask
    cdef Rule rule
    cdef int bitset_idx
    cdef bint has_valid_cats, is_categorical_split
    
    # Leaf node kontrolü
    if nodes[node_id]['is_leaf'] == 1:
        # Leaf node - rule oluştur
        rule = create_rule(len(feature_ranges), tree_index, node_id)
        
        # Numerik constraint'leri ekle
        for i in range(len(feature_ranges)):
            if feature_ranges[i].first != -INFINITY or feature_ranges[i].second != INFINITY:
                update_rule(&rule, i, feature_ranges[i].first, feature_ranges[i].second)
        
        # Kategorik constraint'leri güncelle
        update_cats_for_rule(&rule, cat_mask)
        
        # Leaf değerini ayarla
        raw_value = nodes[node_id]['value']
        rule.value = raw_value * scale_factor
        rule.count = nodes[node_id]['count']
        
        rules.push_back(rule)
        return
    
    # Internal node - split
    bin_feature_idx = nodes[node_id]['feature_idx']  # Bin mapper index
    original_feature_idx = map_bin_to_original_feature(bin_feature_idx, bin_to_original_mapping)  # Orijinal index
    
    left_child = nodes[node_id]['left']
    right_child = nodes[node_id]['right']
    
    # Orijinal feature index'e göre kategorik kontrolü
    is_categorical_split = (is_original_feature_categorical(original_feature_idx, is_categorical_features) and 
                           nodes[node_id]['is_categorical'] == 1)
    
    if is_categorical_split:
        # Kategorik split - bitset kullan
        bitset_idx = nodes[node_id]['bitset_idx']
        
        if (hasattr(predictor, 'raw_left_cat_bitsets') and 
            predictor.raw_left_cat_bitsets is not None and 
            bitset_idx < len(predictor.raw_left_cat_bitsets)):
            
            # Mevcut mask'i kaydet
            prev_mask = cat_mask[original_feature_idx]
            
            # Sol kategorileri bitset'ten al
            left_mask = bitset_to_categorical_mask(
                predictor.raw_left_cat_bitsets[bitset_idx], 
                all_categories_mask[original_feature_idx].size()
            )
            
            # Sol çocuk: left_mask kategorileri
            cat_mask[original_feature_idx] = left_mask
            
            # Mask'in boş olmadığını kontrol et
            has_valid_cats = False
            for i in range(cat_mask[original_feature_idx].size()):
                if cat_mask[original_feature_idx][i] == 1:
                    has_valid_cats = True
                    break
            
            if has_valid_cats:
                traverse_histgb_tree(nodes, left_child, feature_ranges, rules, tree_index,
                                         scale_factor, is_classification, n_classes, class_idx,
                                         cat_mask, is_categorical_features, all_categories_mask, 
                                         predictor, bin_to_original_mapping)
            
            # Sağ çocuk: left_mask'te olmayan kategoriler
            right_mask = get_right_categories_mask(left_mask, all_categories_mask[original_feature_idx])
            cat_mask[original_feature_idx] = right_mask
            
            # Mask'in boş olmadığını kontrol et
            has_valid_cats = False
            for i in range(cat_mask[original_feature_idx].size()):
                if cat_mask[original_feature_idx][i] == 1:
                    has_valid_cats = True
                    break
            
            if has_valid_cats:
                traverse_histgb_tree(nodes, right_child, feature_ranges, rules, tree_index,
                                         scale_factor, is_classification, n_classes, class_idx,
                                         cat_mask, is_categorical_features, all_categories_mask, 
                                         predictor, bin_to_original_mapping)
            
            # Orijinal mask'i geri yükle
            cat_mask[original_feature_idx] = prev_mask
        else:
            # Bitset yok - numerik olarak işle
            threshold = float(nodes[node_id]['num_threshold'])
            prev_range = feature_ranges[original_feature_idx]
            
            # Sol: <= threshold
            feature_ranges[original_feature_idx] = (prev_range.first, threshold)
            traverse_histgb_tree(nodes, left_child, feature_ranges, rules, tree_index,
                                     scale_factor, is_classification, n_classes, class_idx,
                                     cat_mask, is_categorical_features, all_categories_mask, 
                                     predictor, bin_to_original_mapping)
            
            # Sağ: > threshold
            feature_ranges[original_feature_idx] = (threshold, prev_range.second)
            traverse_histgb_tree(nodes, right_child, feature_ranges, rules, tree_index,
                                     scale_factor, is_classification, n_classes, class_idx,
                                     cat_mask, is_categorical_features, all_categories_mask, 
                                     predictor, bin_to_original_mapping)
            
            # Orijinal range'i geri yükle
            feature_ranges[original_feature_idx] = prev_range
    else:
        # Numerik split
        threshold = float(nodes[node_id]['num_threshold'])
        prev_range = feature_ranges[original_feature_idx]
        
        # Sol: <= threshold
        feature_ranges[original_feature_idx] = (prev_range.first, threshold)
        traverse_histgb_tree(nodes, left_child, feature_ranges, rules, tree_index,
                                 scale_factor, is_classification, n_classes, class_idx,
                                 cat_mask, is_categorical_features, all_categories_mask, 
                                 predictor, bin_to_original_mapping)
        
        # Sağ: > threshold
        feature_ranges[original_feature_idx] = (threshold, prev_range.second)
        traverse_histgb_tree(nodes, right_child, feature_ranges, rules, tree_index,
                                 scale_factor, is_classification, n_classes, class_idx,
                                 cat_mask, is_categorical_features, all_categories_mask, 
                                 predictor, bin_to_original_mapping)
        
        # Orijinal range'i geri yükle
        feature_ranges[original_feature_idx] = prev_range


cdef tuple[vector[vector[Rule]], vector[vector[int]], vector[int]] analyze_sklearn(object model, int len_col, int n_classes):
    cdef int tree_index, i, class_idx
    cdef Rule rule
    cdef vector[vector[Rule]] trees = vector[vector[Rule]]()
    cdef vector[Rule] rules
    cdef vector[RangePair] feature_ranges
    cdef double scale_factor = 1.0
    cdef bint is_classification = False
    cdef bint is_histgb = False
    
    # Categorical support variables
    cdef vector[vector[int]] cat_values = vector[vector[int]]()
    cdef vector[int] cat_indices = vector[int]()
    cdef vector[vector[bint]] cat_mask
    cdef vector[bint] is_categorical_features
    cdef vector[vector[bint]] all_categories_mask
    cdef int cat_val, max_cat_val
    
    cat_values.resize(len_col)
    
    cdef object tree
    cdef list trees_list
    cdef object predictor
    cdef vector[int] bin_to_original_mapping
    
    # Determine model type
    model_name = model.__class__.__name__
    is_classification = 'Classifier' in model_name
    is_histgb = 'HistGradientBoosting' in model_name
    

    # Initialize categorical information
    if is_histgb and hasattr(model, '_bin_mapper') and hasattr(model._bin_mapper, 'is_categorical'):
        # HistGradientBoosting - bitsetlerle kategorik desteği
        is_categorical_features = vector[bint](len_col, 0)
        all_categories_mask = vector[vector[bint]]()
        all_categories_mask.resize(len_col)
        
        # DOĞRU YOL: model.is_categorical_ kullan (orijinal feature sırasında)
        if hasattr(model, 'is_categorical_') and model.is_categorical_ is not None:
            # model.is_categorical_ orijinal feature sırasında kategorik bilgiyi verir
            for i in range(min(len(model.is_categorical_), len_col)):
                if model.is_categorical_[i]:
                    is_categorical_features[i] = 1
                    cat_indices.push_back(i)
                    # Başlangıçta boş - gerçek boyut bitset'ten belirlenecek
                    all_categories_mask[i] = vector[bint]()
                else:
                    all_categories_mask[i] = vector[bint]()
            
            # Kategorik özellikler için bitsetlerden bilgi al
            if (hasattr(model._bin_mapper, 'make_known_categories_bitsets') and 
                callable(getattr(model._bin_mapper, 'make_known_categories_bitsets'))):
                
                try:
                    known_categories_bitsets, _ = model._bin_mapper.make_known_categories_bitsets()

                    # Orijinal feature'lardan kategorik olanların indekslerini bul
                    original_categorical_indices = []
                    for i in range(len(model.is_categorical_)):
                        if model.is_categorical_[i]:
                            original_categorical_indices.append(i)
                    
                    # FIXED: Bitsetleri orijinal indekslerle eşleştir
                    for bitset_idx in range(min(len(known_categories_bitsets), len(original_categorical_indices))):
                        original_feature_idx = original_categorical_indices[bitset_idx]
                        
                        if original_feature_idx < len_col:
                            # Bitset'in boyutunu dinamik olarak belirle
                            bitset_data = known_categories_bitsets[bitset_idx]
                            max_possible_categories = len(bitset_data) * 32  # Her word 32 bit
                            
                            # Bu özellik için tüm kategorileri bitset'ten al
                            all_categories_mask[original_feature_idx] = bitset_to_categorical_mask(
                                bitset_data, 
                                max_possible_categories
                            )
                            
                            # cat_values'u güncelle - sadece gerçekte var olan kategoriler
                            for cat_val in range(all_categories_mask[original_feature_idx].size()):
                                if all_categories_mask[original_feature_idx][cat_val] == 1:
                                    cat_values[original_feature_idx].push_back(cat_val)
                                
                except Exception as e:
                    print(f"Bitset extraction failed: {e}")
                    # Fallback - kategorik yok
                    is_categorical_features = vector[bint](len_col, 0)
                    all_categories_mask = vector[vector[bint]]()
                    all_categories_mask.resize(len_col)
                    cat_indices.clear()
            else:
                # make_known_categories_bitsets yok - tüm mask'leri boş bırak
                # Kategorik özellikler için bile boş mask (bitset bilgisi yok)
                pass
        else:
            # model.is_categorical_ yok - kategorik yok
            is_categorical_features = vector[bint](len_col, 0)
            all_categories_mask = vector[vector[bint]]()
            all_categories_mask.resize(len_col)
    else:
        # Standard sklearn models - kategorik yok
        is_categorical_features = vector[bint](len_col, 0)
        all_categories_mask = vector[vector[bint]]()
        all_categories_mask.resize(len_col)
        
    # Extract trees from model
    if is_histgb:
        # HistGradientBoosting models
        if not hasattr(model, '_predictors'):
            raise ValueError("HistGradientBoosting model does not have _predictors attribute")
        
        predictors = model._predictors
        if not predictors:
            raise ValueError("No predictors found in HistGradientBoosting model")
        
        scale_factor = getattr(model, 'learning_rate', 1.0)

        bin_to_original_mapping = create_bin_to_original_mapping(model, len_col)
        
        # Process each predictor (tree)
        for tree_index in range(len(predictors)):
            predictor_group = predictors[tree_index]
            
            # Handle different class structures
            if isinstance(predictor_group, list):
                # Multiple classes
                for class_idx in range(len(predictor_group)):
                    if class_idx < n_classes:
                        predictor = predictor_group[class_idx]
                        
                        # Initialize feature ranges
                        feature_ranges = vector[RangePair](len_col)
                        for i in range(len_col):
                            feature_ranges[i] = (-INFINITY, INFINITY)
                        
                        # Initialize categorical mask
                        cat_mask.clear()
                        cat_mask.resize(len_col)
                        for i in range(len_col):
                            cat_mask[i] = all_categories_mask[i]  # Tüm kategoriler başlangıçta aktif
                        
                        rules = vector[Rule]()
                        effective_tree_index = tree_index * n_classes + class_idx
                        
                        # Traverse the tree
                        traverse_histgb_tree(predictor.nodes, 0, feature_ranges, rules, 
                                           effective_tree_index, scale_factor, is_classification, 
                                           n_classes, class_idx, cat_mask, is_categorical_features, 
                                           all_categories_mask, predictor, bin_to_original_mapping)
                        
                        if rules.empty():
                            raise ValueError(f"No rules extracted from HistGradientBoosting tree {tree_index}, class {class_idx}")
                        
                        sort(rules.begin(), rules.end(), compare_rules)
                        trees.push_back(rules)
            else:
                # Single class/regression
                predictor = predictor_group
                
                # Initialize feature ranges
                feature_ranges = vector[RangePair](len_col)
                for i in range(len_col):
                    feature_ranges[i] = (-INFINITY, INFINITY)
                
                # Initialize categorical mask
                cat_mask.clear()
                cat_mask.resize(len_col)
                for i in range(len_col):
                    cat_mask[i] = all_categories_mask[i]  # Tüm kategoriler başlangıçta aktif
                
                rules = vector[Rule]()
                
                # Traverse the tree
                traverse_histgb_tree(predictor.nodes, 0, feature_ranges, rules, 
                                   tree_index, scale_factor, is_classification, 
                                   n_classes, 0, cat_mask, is_categorical_features, 
                                   all_categories_mask, predictor, bin_to_original_mapping)
                
                if rules.empty():
                    raise ValueError(f"No rules extracted from HistGradientBoosting tree {tree_index}")
                
                sort(rules.begin(), rules.end(), compare_rules)
                trees.push_back(rules)
    
    elif hasattr(model, 'estimators_'):
        # Standard ensemble methods
        if 'RandomForest' in model_name or 'ExtraTrees' in model_name:
            trees_list = [est.tree_ for est in model.estimators_]
            scale_factor = 1.0 / len(trees_list)
        elif 'GradientBoosting' in model_name:
            trees_list = []
            for i in range(len(model.estimators_)):
                tree_est = model.estimators_[i, 0]
                trees_list.append(tree_est.tree_)
            scale_factor = model.learning_rate
        elif 'AdaBoost' in model_name:
            trees_list = [est.tree_ for est in model.estimators_]
            scale_factor = 1.0
        else:
            trees_list = [est.tree_ for est in model.estimators_]
            scale_factor = 1.0 / len(trees_list)
        
        # Process each tree using standard sklearn approach
        for tree_index in range(len(trees_list)):
            tree = trees_list[tree_index]
            
            if not hasattr(tree, 'children_left') or not hasattr(tree, 'children_right'):
                raise ValueError(f"Invalid tree structure at index {tree_index}")
            
            if len(tree.children_left) == 0:
                raise ValueError(f"Empty tree at index {tree_index}")
            
            # For multiclass, create n_classes copies of each tree
            if is_classification and n_classes > 2:
                for class_idx in range(n_classes):
                    feature_ranges = vector[RangePair](len_col)
                    for i in range(len_col):
                        feature_ranges[i] = (-INFINITY, INFINITY)
                    
                    rules = vector[Rule]()
                    effective_tree_index = tree_index * n_classes + class_idx
                    
                    traverse_sklearn_tree(tree, 0, feature_ranges, rules, effective_tree_index, scale_factor, is_classification, n_classes, class_idx)
                    
                    if rules.empty():
                        raise ValueError(f"No rules extracted from tree {tree_index}, class {class_idx}")
                    
                    sort(rules.begin(), rules.end(), compare_rules)
                    trees.push_back(rules)
            else:
                # Binary classification or regression
                feature_ranges = vector[RangePair](len_col)
                for i in range(len_col):
                    feature_ranges[i] = (-INFINITY, INFINITY)
                
                rules = vector[Rule]()
                
                traverse_sklearn_tree(tree, 0, feature_ranges, rules, tree_index, scale_factor, is_classification, n_classes, 0)
                
                if rules.empty():
                    raise ValueError(f"No rules extracted from tree {tree_index}")
                
                sort(rules.begin(), rules.end(), compare_rules)
                trees.push_back(rules)
    
    else:
        # Single tree
        if hasattr(model, 'tree_'):
            trees_list = [model.tree_]
        else:
            raise ValueError(f"Cannot extract tree from model type: {type(model)}")
        scale_factor = 1.0
        
        tree = trees_list[0]
        
        feature_ranges = vector[RangePair](len_col)
        for i in range(len_col):
            feature_ranges[i] = (-INFINITY, INFINITY)
        
        rules = vector[Rule]()
        
        traverse_sklearn_tree(tree, 0, feature_ranges, rules, 0, scale_factor, is_classification, n_classes, 0)
        
        if rules.empty():
            raise ValueError("No rules extracted from single tree")
        
        sort(rules.begin(), rules.end(), compare_rules)
        trees.push_back(rules)
    
    # Validate we have trees
    if trees.empty():
        raise ValueError("No trees found in model")
    
    # Return the results
    return trees, cat_values, cat_indices