cdef dict count_tree_indices(cnp.ndarray[cnp.int32_t, ndim=2] array):
    cdef Py_ssize_t num_trees = array.shape[1]
    cdef Py_ssize_t num_rows = array.shape[0]
    cdef dict results = {}
    cdef dict tree_results
    cdef Py_ssize_t tree_idx, row_idx
    cdef int val

    for tree_idx in range(num_trees):
        tree_results = {}
        for row_idx in range(num_rows):
            val = array[row_idx, tree_idx]
            if val != 0:
                if val in tree_results:
                    tree_results[val] = tree_results[val] + 1
                else:
                    tree_results[val] = 1
        results[tree_idx] = tree_results

    return results


cdef vector[vector[Rule]] update_leaf_counts(vector[vector[Rule]] trees, object model, object back_data, str model_type):
    cdef:
        cnp.ndarray[cnp.int32_t, ndim=2] back_data_
        dict dict_counts
        object back_data_d_matrix

        # Variables for looping
        size_t num_trees = trees.size()
        size_t tree_size
        Rule* rule_ptr
        vector[Rule]* tree_ptr

        size_t tree_index, leaf_index
        float leaf_counts
        dict node_counts

    if model_type == "xgboost":
        back_data_d_matrix = convert_d_matrix(back_data)
        back_data_ = model.predict(back_data_d_matrix, pred_leaf=True).astype(np.int32)
        back_data_ = xgb_leaf_correction(trees, back_data_)

    elif model_type == "catboost":
        back_data_ = model.calc_leaf_indexes(back_data).astype(np.int32)

    elif model_type == "lightgbm":
        back_data_ = model.predict(back_data, pred_leaf=True).astype(np.int32)

    dict_counts = count_tree_indices(back_data_)

    for tree_index in range(num_trees):
        tree_ptr = &trees[tree_index]
        tree_size = tree_ptr[0].size()
        for leaf_index in range(tree_size):
            rule_ptr = &(tree_ptr[0][leaf_index])
            rule_ptr.count = 0.0

    for tree_index, node_counts in dict_counts.items():
        for leaf_index, leaf_counts in node_counts.items():
            tree_ptr = &trees[tree_index]
            rule_ptr = &(tree_ptr[0][leaf_index])
            rule_ptr.count = leaf_counts

    return trees
