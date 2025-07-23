from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.math cimport INFINITY
from libcpp.algorithm cimport sort

# Expose the Rule helpers you already have
from .rule cimport Rule, create_rule, update_rule, compare_rules, update_cats_for_rule

ctypedef pair[float, float] RangePair

############################################################
# Helper utilities                                          #
############################################################

cdef inline bint _is_categorical_node(dict node):
    """Return True if *node* performs a categorical split."""
    return (("left_cats" in node and node["left_cats"]) or
            ("right_cats" in node and node["right_cats"]))


############################################################
# Tree traversal                                            #
############################################################

cdef void traverse_perpetual_tree(dict nodes,
                                  int node_id,
                                  vector[RangePair]& feature_ranges,
                                  vector[Rule]& rules,
                                  int tree_index,
                                  vector[vector[bint]]& cat_mask):
    """Depth‑first traversal that converts a Perpetual tree into Rule objects."""

    cdef str node_key = str(node_id)
    if node_key not in nodes:
        return  # Safety check
    cdef dict node = nodes[node_key]
    cdef int feature_index, i, cat_idx
    cdef float threshold
    cdef RangePair prev_range
    cdef vector[bint] prev_mask
    cdef Rule rule
    cdef list left_cats 
    # -------------------------------------------------------
    # Leaf node                                             
    # -------------------------------------------------------
    if node.get("is_leaf", False):
        rule = create_rule(len(feature_ranges), tree_index, node_id)

        # numerical ranges
        for i in range(len(feature_ranges)):
            if (feature_ranges[i].first != -INFINITY or
                feature_ranges[i].second != INFINITY):
                update_rule(&rule, i,
                            feature_ranges[i].first,
                            feature_ranges[i].second)

        # categorical masks
        update_cats_for_rule(&rule, cat_mask)

        rule.value = node.get("weight_value", 0.0)
        rule.count = node.get("hessian_sum", 0.0)
        rules.push_back(rule)
        return

    # -------------------------------------------------------
    # Split node                                            
    # -------------------------------------------------------
    if "split_feature" not in node:
        return  # Safety check
    feature_index = <int>node["split_feature"]

    if _is_categorical_node(node):
        # ---------------------------------------------------
        # Categorical split – categories listed in left_cats
        # ---------------------------------------------------
        left_cats = node.get("left_cats", [])

        # ‑‑ backup current mask
        prev_mask = cat_mask[feature_index]

        # LEFT branch: keep only categories in *left_cats*
        for i in range(prev_mask.size()):
            cat_mask[feature_index][i] = 0
        for cat_idx in left_cats:
            if 0 <= cat_idx < prev_mask.size() and prev_mask[cat_idx]:
                cat_mask[feature_index][cat_idx] = 1

        if "left_child" in node:
            traverse_perpetual_tree(nodes, node["left_child"],
                                     feature_ranges, rules,
                                     tree_index, cat_mask)

        # RIGHT branch: categories not in *left_cats*
        cat_mask[feature_index] = prev_mask               # restore
        for cat_idx in left_cats:
            if 0 <= cat_idx < cat_mask[feature_index].size():
                cat_mask[feature_index][cat_idx] = 0

        if "right_child" in node:
            traverse_perpetual_tree(nodes, node["right_child"],
                                     feature_ranges, rules,
                                     tree_index, cat_mask)

        cat_mask[feature_index] = prev_mask               # full restore

    else:
        # ---------------------------------------------------
        # Numerical split                                   
        # ---------------------------------------------------
        if "split_value" not in node:
            return  # Safety check
        threshold = node["split_value"]
        prev_range = feature_ranges[feature_index]

        # LEFT: < threshold
        feature_ranges[feature_index] = (prev_range.first, threshold)
        if "left_child" in node:
            traverse_perpetual_tree(nodes, node["left_child"],
                                     feature_ranges, rules,
                                     tree_index, cat_mask)

        # RIGHT: >= threshold
        feature_ranges[feature_index] = (threshold, prev_range.second)
        if "right_child" in node:
            traverse_perpetual_tree(nodes, node["right_child"],
                                     feature_ranges, rules,
                                     tree_index, cat_mask)

        feature_ranges[feature_index] = prev_range         # restore


############################################################
# Public API                                               #
############################################################

cdef tuple[vector[vector[Rule]], vector[vector[int]], vector[int]] analyze_perpetual(object model):
    """
    Convert a *Perpetual* gradient‑boosted decision‑tree model to a rule list.

    Works for all objectives:
    * Regression / binary classification → top‑level key ``"trees"``
    * Multiclass classification          → top‑level key ``"boosters"`` where
      each booster corresponds to one class and owns its own ``"trees"`` list.

    Returns
    -------
    (all_trees, cat_values, cat_indices)
        all_trees   : vector[vector[Rule]]  – rules grouped per tree
        cat_values  : vector[vector[int]]   – allowed category ids per feature
        cat_indices : vector[int]           – indices of categorical features
    """

    import json

    cdef dict dump = (model if isinstance(model, dict)
                      else json.loads(model.json_dump()))
    cdef dict meta = dump.get("metadata", {})

    cdef list feature_names = json.loads(meta.get("feature_names_in_", "[]"))
    cdef dict cat_mapping = json.loads(meta.get("cat_mapping", "{}"))
    
    # Fix the type casting issue
    cdef int n_features
    try:
        n_features_str = meta.get("n_features_", "0")
        if isinstance(n_features_str, str):
            n_features = int(n_features_str)
        else:
            n_features = <int>n_features_str
    except (ValueError, TypeError):
        n_features = len(feature_names)
    
    if n_features == 0:
        n_features = len(feature_names)

    # -------------------------------------------------------
    # Build categorical helpers                             
    # -------------------------------------------------------
    cdef vector[vector[int]] cat_values = vector[vector[int]]()
    cdef vector[int]        cat_indices = vector[int]()
    cdef int i, j

    cat_values.resize(n_features)
    for i, fname in enumerate(feature_names):
        if fname in cat_mapping:
            cat_indices.push_back(i)
            for j in range(len(cat_mapping[fname])):
                cat_values[i].push_back(j)   # simply enumerate categories

    # Template cat_mask (all categories allowed)
    cdef vector[vector[bint]] template_mask
    template_mask.resize(n_features)
    for i in range(n_features):
        template_mask[i] = vector[bint](cat_values[i].size(), 1)

    # -------------------------------------------------------
    # Iterate over every tree in the model                  
    # -------------------------------------------------------
    cdef vector[vector[Rule]] all_trees = vector[vector[Rule]]()
    cdef vector[RangePair] feature_ranges
    cdef vector[Rule] rules
    cdef int tree_idx = 0
    cdef vector[vector[bint]] cat_mask
    cdef int root_id
    
    def _tree_iter():
        if "boosters" in dump:              # multiclass
            for booster in dump["boosters"]:
                for t in booster.get("trees", []):
                    yield t
        elif "trees" in dump:               # single class / regression
            for t in dump["trees"]:
                yield t
        else:
            raise ValueError("Model dump has neither 'trees' nor 'boosters'.")

    for tree in _tree_iter():
        if "nodes" not in tree:
            continue  # Skip malformed trees
        nodes = tree["nodes"]

        # reset feature ranges and masks
        feature_ranges = vector[RangePair](n_features)
        for i in range(n_features):
            feature_ranges[i] = (-INFINITY, INFINITY)

        # deep‑copy template mask for this tree
        cat_mask.resize(n_features)
        for i in range(n_features):
            cat_mask[i] = vector[bint](template_mask[i])

        rules = vector[Rule]()
        
        # Find root node (could be "0" or have node_type="Root")
        root_id = 0
        if "0" not in nodes:
            # Try to find root by node_type
            for node_key, node_data in nodes.items():
                if node_data.get("node_type") == "Root":
                    root_id = int(node_key)
                    break
        
        traverse_perpetual_tree(nodes, root_id,
                                feature_ranges, rules,
                                tree_idx, cat_mask)

        sort(rules.begin(), rules.end(), compare_rules)
        all_trees.push_back(rules)
        tree_idx += 1

    return all_trees, cat_values, cat_indices