from .rule cimport create_rule, update_rule, compare_rules
from libc.math cimport INFINITY, log, exp
from libcpp.algorithm cimport sort
from libcpp.pair cimport pair

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

    # Standard sklearn leaf check: children_left == children_right (both -1 for leaves)
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
                # Multiclass: tree.value[node_id] shape is (1, n_classes)
                # Use the value for the specific class
                if tree.value[node_id].shape[1] > class_idx:
                    class_count = tree.value[node_id][0][class_idx]
                    total_samples = tree.n_node_samples[node_id]
                    prob = class_count / total_samples if total_samples > 0 else (1.0 / n_classes)
                    
                    # Convert to logit space then scale
                    logit_value = convert_sklearn_to_logit(prob)
                    rule.value = logit_value * scale_factor
                else:
                    # Fallback
                    rule.value = 0.0
            else:
                # Binary classification: tree.value[node_id] shape is (1, 2)
                # [0] = negative class count, [1] = positive class count
                if tree.value[node_id].shape[1] >= 2:
                    positive_count = tree.value[node_id][0][1]
                    total_samples = tree.n_node_samples[node_id]
                    prob = positive_count / total_samples if total_samples > 0 else 0.5
                    
                    # Convert to logit space then scale
                    logit_value = convert_sklearn_to_logit(prob)
                    rule.value = logit_value * scale_factor
                else:
                    # Single class case
                    raw_value = tree.value[node_id][0][0]
                    rule.value = raw_value * scale_factor
        else:
            # Regression: use mean prediction
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


cdef tuple[vector[vector[Rule]], vector[vector[int]], vector[int]] analyze_sklearn(object model, int len_col, int n_classes):
    cdef int tree_index, i, class_idx
    cdef Rule rule
    cdef vector[vector[Rule]] trees = vector[vector[Rule]]()
    cdef vector[Rule] rules
    cdef vector[RangePair] feature_ranges
    cdef double scale_factor = 1.0
    cdef bint is_classification = False
    
    # Mock categorical placeholders (empty for sklearn)
    cdef vector[vector[int]] cat_values = vector[vector[int]]()
    cdef vector[int] cat_indices = vector[int]()
    cat_values.resize(len_col)
    
    cdef object tree
    cdef list trees_list
    
    # Determine model type
    model_name = model.__class__.__name__
    is_classification = 'Classifier' in model_name
    
    # Extract trees from model
    if hasattr(model, 'estimators_'):
        # Ensemble methods
        if 'RandomForest' in model_name or 'ExtraTrees' in model_name:
            trees_list = [est.tree_ for est in model.estimators_]
            scale_factor = 1.0 / len(trees_list)  # Average predictions
        elif 'GradientBoosting' in model_name:
            trees_list = []
            for i in range(len(model.estimators_)):
                tree_est = model.estimators_[i, 0]  # First (and only) class
                trees_list.append(tree_est.tree_)
            scale_factor = model.learning_rate  # Apply learning rate
        elif 'AdaBoost' in model_name:
            trees_list = [est.tree_ for est in model.estimators_]
            scale_factor = 1.0  # Weights handled separately
        else:
            # Generic ensemble handling
            trees_list = [est.tree_ for est in model.estimators_]
            scale_factor = 1.0 / len(trees_list)
    else:
        # Single tree
        if hasattr(model, 'tree_'):
            trees_list = [model.tree_]
        else:
            raise ValueError(f"Cannot extract tree from model type: {type(model)}")
        scale_factor = 1.0
    
    # Validate we have trees
    if not trees_list:
        raise ValueError("No trees found in model")
    
    # Process each tree
    for tree_index in range(len(trees_list)):
        tree = trees_list[tree_index]
        
        # Validate tree structure
        if not hasattr(tree, 'children_left') or not hasattr(tree, 'children_right'):
            raise ValueError(f"Invalid tree structure at index {tree_index}")
        
        if len(tree.children_left) == 0:
            raise ValueError(f"Empty tree at index {tree_index}")
        
        # For multiclass, create n_classes copies of each tree
        if is_classification and n_classes > 2:
            # Create rules for each class
            for class_idx in range(n_classes):
                # Initialize feature ranges to unbounded
                feature_ranges = vector[RangePair](len_col)
                for i in range(len_col):
                    feature_ranges[i] = (-INFINITY, INFINITY)
                
                # Initialize rules vector for this tree-class combination
                rules = vector[Rule]()
                
                # Calculate the effective tree index (tree_index * n_classes + class_idx)
                effective_tree_index = tree_index * n_classes + class_idx
                
                # Traverse tree starting from root (node 0)
                traverse_sklearn_tree(tree, 0, feature_ranges, rules, effective_tree_index, scale_factor, is_classification, n_classes, class_idx)
                
                # Ensure we have rules
                if rules.empty():
                    raise ValueError(f"No rules extracted from tree {tree_index}, class {class_idx}")
                
                # Sort rules by some criteria
                sort(rules.begin(), rules.end(), compare_rules)
                
                # Add this tree-class combination's rules to the collection
                trees.push_back(rules)
        else:
            # Binary classification or regression - single tree
            # Initialize feature ranges to unbounded
            feature_ranges = vector[RangePair](len_col)
            for i in range(len_col):
                feature_ranges[i] = (-INFINITY, INFINITY)
            
            # Initialize rules vector for this tree
            rules = vector[Rule]()
            
            # Traverse tree starting from root (node 0)
            traverse_sklearn_tree(tree, 0, feature_ranges, rules, tree_index, scale_factor, is_classification, n_classes, 0)
            
            # Ensure we have rules
            if rules.empty():
                raise ValueError(f"No rules extracted from tree {tree_index}")
            
            # Sort rules by some criteria
            sort(rules.begin(), rules.end(), compare_rules)
            
            # Add this tree's rules to the collection
            trees.push_back(rules)
    
    # Return the results
    return trees, cat_values, cat_indices