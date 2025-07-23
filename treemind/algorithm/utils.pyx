from libcpp.vector cimport vector
from .rule cimport get_split_point, check_value

from libc.math cimport sqrt

import numpy as np
cimport numpy as cnp
from pandas import Categorical


cdef inline float cmax(float a, float b) noexcept nogil:
    return a if a > b else b


cdef inline float cmin(float a, float b) noexcept nogil:
    return a if a < b else b


cdef object replace_inf(object data, str column_name):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] unique_points_ = np.asarray(data[column_name].unique(), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] unique_points = np.sort(unique_points_[np.logical_not(np.isinf(unique_points_))])
    cdef float max_main = unique_points.max()
    cdef float difference_main = max_main - (unique_points[unique_points.size -2] if unique_points.size > 1 else max_main * 0.9)

    data.loc[np.isinf(data[column_name]), column_name] = max_main + difference_main

    return data


cdef add_lower_bound(object data, int loc, str column):
    cdef:
        str column_ub = f"{column}_ub"
        cnp.ndarray[cnp.float32_t, ndim=1] unique_values = np.sort(data.loc[:, column_ub].unique())
        cnp.ndarray[cnp.float32_t, ndim=1] lower_bounds = np.empty_like(unique_values, dtype=np.float32)

    lower_bounds[0] = -np.inf
    lower_bounds[1:] = unique_values[:unique_values.size - 1]

    categories = Categorical(data[column_ub], categories=unique_values, ordered=True)

    data.insert(loc, f"{column}_lb", lower_bounds[categories.codes])


cdef vector[vector[Rule]] filter_class_trees(const vector[vector[Rule]]& trees,  
                                              const int n_classes, 
                                              const int class_idx):
    cdef vector[vector[Rule]] class_trees = vector[vector[Rule]]()
    cdef vector[Rule] class_tree
    cdef size_t i
    cdef int tree_idx

    # Her sınıfa ait ağaç sayısı = toplam ağaç / sınıf sayısı
    class_trees.resize(trees.size() // n_classes)

    for i in range(class_trees.size()):
        tree_idx = i * n_classes + class_idx
        class_trees[i] = trees[tree_idx]

    return class_trees


cdef tuple[vector[vector[float]],
        vector[float],
        vector[float],
        vector[float]] _analyze_feature(const vector[vector[Rule]]& trees,
                                                const vector[int]& columns,
                                                const vector[vector[int]]& cat_cols):
    cdef:
        vector[vector[float]] split_points_list = vector[vector[float]]()
        size_t num_cols = columns.size()
        size_t max_size = 1
        size_t num_trees = trees.size()
        size_t i, j, k, idx, tree_idx, rule_idx

        # Statistics vectors
        vector[float] average_counts
        vector[float] mean_values
        vector[float] ensemble_std

        # Per-tree temporary storage
        vector[float] tree_count
        vector[float] tree_sum
        vector[float] tree_squared_sum

        vector[vector[size_t]] valid_indices_per_col = vector[vector[size_t]](num_cols)
        vector[size_t] multi_index = vector[size_t](num_cols)
        vector[size_t] sizes = vector[size_t](num_cols)

        float rule_value, rule_count, squared_value
        float mean, variance, std, count
        bint skip_rule
        const Rule* rule_ptr
        const vector[Rule]* tree_ptr

        cdef bint done

        vector[vector[float]] combined_points
        size_t col_idx, remaining, point_idx, split_size
        float val
        int val_idx
        vector[vector[int]] cat_slice = vector[vector[int]]()

        vector[bint] is_categorical = vector[bint](num_cols)
        
        size_t total_split_points = 0

    # Pre-compute categorical flags and prepare data
    for i in range(columns.size()):
        col_idx = columns[i]
        cat_slice.push_back(cat_cols[col_idx])
        is_categorical[i] = (cat_slice[i].size() > 0)

    for col_idx in range(num_cols):
        if not is_categorical[col_idx]:  # NUMERICAL
            split_points = get_split_point(trees, columns[col_idx])
        else:  # CATEGORICAL
            split_points = vector[float]()
            split_points.reserve(cat_slice[col_idx].size())  # Only for categorical
            for i in range(cat_slice[col_idx].size()):
                split_points.push_back(cat_slice[col_idx][i])

        split_points_list.push_back(split_points)
        sizes[col_idx] = split_points.size()
        total_split_points += split_points.size()
        max_size *= sizes[col_idx]
        
        valid_indices_per_col[col_idx].reserve(split_points.size())

    # Initialize result vectors
    average_counts = vector[float](max_size, 0.0)
    mean_values = vector[float](max_size, 0.0)
    ensemble_std = vector[float](max_size, 0.0)

    # Initialize temporary vectors
    tree_count = vector[float](max_size)
    tree_sum = vector[float](max_size)
    tree_squared_sum = vector[float](max_size)

    with nogil:
        # Process each tree
        for tree_idx in range(num_trees):
            tree_ptr = &trees[tree_idx]
            
            # Reset per-tree accumulators (more efficient than assign)
            for idx in range(max_size):
                tree_sum[idx] = 0.0
                tree_squared_sum[idx] = 0.0
                tree_count[idx] = 0.0
            
            # Process each rule in the current tree
            for rule_idx in range(tree_ptr.size()):
                rule_ptr = &(tree_ptr[0][rule_idx])
                rule_value = rule_ptr.value
                rule_count = rule_ptr.count
                squared_value = rule_value * rule_value

                skip_rule = False
                
                for k in range(num_cols):
                    valid_indices_per_col[k].clear()
                    split_size = split_points_list[k].size()
                    
                    # Adaptive capacity - only grow if needed
                    if valid_indices_per_col[k].capacity() < split_size:
                        valid_indices_per_col[k].reserve(split_size)
                    
                    if not is_categorical[k]:  # NUMERICAL - optimized branch
                        for i in range(split_size):
                            val = split_points_list[k][i]
                            if check_value(rule_ptr, columns[k], val):
                                valid_indices_per_col[k].push_back(i)
                    else:  # CATEGORICAL - optimized branch
                        for i in range(split_size):
                            val_idx = <int>split_points_list[k][i]
                            if rule_ptr.cats[columns[k]][val_idx]:
                                valid_indices_per_col[k].push_back(i)

                    if valid_indices_per_col[k].empty():
                        skip_rule = True
                        break
                        
                if skip_rule:
                    continue

                if num_cols == 1:
                    # Special case for single column - avoid multi-index overhead
                    for i in range(valid_indices_per_col[0].size()):
                        idx = valid_indices_per_col[0][i]
                        tree_sum[idx] += rule_value * rule_count
                        tree_squared_sum[idx] += squared_value * rule_count
                        tree_count[idx] += rule_count
                elif num_cols == 2:
                    # Special case for two columns - common case optimization
                    for i in range(valid_indices_per_col[0].size()):
                        for j in range(valid_indices_per_col[1].size()):
                            idx = valid_indices_per_col[0][i] * sizes[1] + valid_indices_per_col[1][j]
                            tree_sum[idx] += rule_value * rule_count
                            tree_squared_sum[idx] += squared_value * rule_count
                            tree_count[idx] += rule_count

                elif num_cols == 3:
                    # Özel 3 sütun optimizasyonu - nested loop'lar
                    for i in range(valid_indices_per_col[0].size()):
                        for j in range(valid_indices_per_col[1].size()):
                            for k in range(valid_indices_per_col[2].size()):
                                idx = (valid_indices_per_col[0][i] * sizes[1] + valid_indices_per_col[1][j]) * sizes[2] + valid_indices_per_col[2][k]
                                
                                tree_sum[idx] += rule_value * rule_count
                                tree_squared_sum[idx] += squared_value * rule_count
                                tree_count[idx] += rule_count

                elif num_cols == 4:
                    # Özel 4 sütun optimizasyonu
                    for i in range(valid_indices_per_col[0].size()):
                        for j in range(valid_indices_per_col[1].size()):
                            for k in range(valid_indices_per_col[2].size()):
                                for l in range(valid_indices_per_col[3].size()):
                                    idx = ((valid_indices_per_col[0][i] * sizes[1] + valid_indices_per_col[1][j]) * sizes[2] + valid_indices_per_col[2][k]) * sizes[3] + valid_indices_per_col[3][l]

                                    tree_sum[idx] += rule_value * rule_count
                                    tree_squared_sum[idx] += squared_value * rule_count
                                    tree_count[idx] += rule_count

                else:
                    # General case - optimized multi-index iteration
                    multi_index.assign(num_cols, 0)
                    done = False
                    while not done:
                        idx = valid_indices_per_col[0][multi_index[0]]
                        for k in range(1, num_cols):
                            idx = idx * sizes[k] + valid_indices_per_col[k][multi_index[k]]
        
                        # Update statistics
                        tree_sum[idx] += rule_value * rule_count
                        tree_squared_sum[idx] += squared_value * rule_count
                        tree_count[idx] += rule_count

                        k = num_cols - 1
                        while k >= 0:
                            multi_index[k] += 1
                            if multi_index[k] < valid_indices_per_col[k].size():
                                break
                            multi_index[k] = 0
                            k -= 1
                        if k < 0:
                            done = True

            for idx in range(max_size):
                count = tree_count[idx]
                if count > 0:
                    average_counts[idx] += count
                    mean = tree_sum[idx] / count
                    
                    # More numerically stable variance calculation
                    variance = (tree_squared_sum[idx] - tree_sum[idx] * mean) / count
                    if variance < 0.0:
                        variance = 0.0
                    std = sqrt(variance)
                    
                    ensemble_std[idx] += std
                    mean_values[idx] += mean

        # Finalize aggregate statistics
        for idx in range(max_size):
            average_counts[idx] /= num_trees

    combined_points = vector[vector[float]](max_size)
    for idx in range(max_size):
        combined_points[idx] = vector[float](num_cols)
        remaining = idx
        for k in range(num_cols - 1, -1, -1):
            point_idx = remaining % sizes[k]
            remaining //= sizes[k]
            combined_points[idx][k] = split_points_list[k][point_idx]

    return combined_points, mean_values, ensemble_std, average_counts