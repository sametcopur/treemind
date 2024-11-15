from libcpp.vector cimport vector
from .rule cimport filter_trees, get_split_point, check_value

from libc.math cimport sqrt

from cython cimport boundscheck, wraparound, initializedcheck, nonecheck, cdivision, overflowcheck, infer_types

import numpy as np
cimport numpy as cnp
from pandas import Categorical

@boundscheck(False)
@nonecheck(False)
@wraparound(False)
@initializedcheck(False)
@cdivision(True)
@overflowcheck(False)
@infer_types(True)
cdef inline double cmax(double a, double b) noexcept nogil:
    return a if a > b else b

@boundscheck(False)
@nonecheck(False)
@wraparound(False)
@initializedcheck(False)
@cdivision(True)
@overflowcheck(False)
@infer_types(True)
cdef inline double cmin(double a, double b) noexcept nogil:
    return a if a < b else b


@infer_types(True)
cdef object replace_inf(object data, str column_name):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] unique_points_ = np.asarray(data[column_name].unique(), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] unique_points = np.sort(unique_points_[np.logical_not(np.isinf(unique_points_))])
    cdef double max_main = unique_points.max()
    cdef double difference_main = max_main - (unique_points[-2] if unique_points.size > 1 else max_main * 0.9)

    data.loc[np.isinf(data[column_name]), column_name] = max_main + difference_main

    return data


@infer_types(True)
cdef add_lower_bound(object data, int loc, str column):
    cdef:
        str column_ub = f"{column}_ub"
        cnp.ndarray[cnp.float64_t, ndim=1] unique_values = np.sort(data.loc[:, column_ub].unique())
        cnp.ndarray[cnp.float64_t, ndim=1] lower_bounds = np.empty_like(unique_values, dtype=np.float64)

    lower_bounds[0] = -np.inf
    lower_bounds[1:] = unique_values[:-1]

    categories = Categorical(data[column_ub], categories=unique_values, ordered=True)

    data.insert(loc, f"{column}_lb", lower_bounds[categories.codes])


@boundscheck(False)
@nonecheck(False)
@wraparound(False)
@initializedcheck(False)
@cdivision(True)
@overflowcheck(False)
@infer_types(True)
cdef tuple[vector[vector[double]],
           vector[double],
           vector[double],
           vector[double]] _analyze_feature(const vector[vector[Rule]]& trees,
                                                    const vector[int]& columns):
    cdef:
        vector[vector[Rule]] filtered_trees = filter_trees(trees, columns)
        vector[vector[double]] split_points_list = vector[vector[double]]()
        size_t num_cols = columns.size()
        size_t max_size = 1
        size_t num_trees = filtered_trees.size()
        size_t i, j, k, idx, tree_idx, rule_idx

        # Statistics vectors
        vector[double] average_counts
        vector[double] mean_values
        vector[double] total_tree_counts
        vector[double] ensemble_std

        # Per-tree temporary storage
        vector[double] tree_count
        vector[double] tree_sum
        vector[double] tree_squared_sum

        # Index handling
        vector[vector[size_t]] valid_indices_per_col = vector[vector[size_t]](num_cols)
        vector[size_t] multi_index = vector[size_t](num_cols)
        vector[size_t] sizes = vector[size_t](num_cols)

        double rule_value, rule_count, squared_value
        double mean, variance, std, count
        bint skip_rule
        const Rule* rule_ptr
        const vector[Rule]* tree_ptr

        cdef bint done

        vector[vector[double]] combined_points
        size_t total_points
        size_t col_idx, remaining, point_idx, split_size

    # Initialize split points and calculate max_size
    for col_idx in range(num_cols):
        split_points = get_split_point(filtered_trees, columns[col_idx])
        split_points_list.push_back(split_points)
        sizes[col_idx] = split_points.size()
        max_size *= sizes[col_idx]

    # Initialize result vectors
    average_counts = vector[double](max_size, 0.0)
    mean_values = vector[double](max_size, 0.0)
    total_tree_counts = vector[double](max_size, 0.0)
    ensemble_std = vector[double](max_size, 0.0)

    # Initialize temporary vectors
    tree_count = vector[double](max_size)
    tree_sum = vector[double](max_size)
    tree_squared_sum = vector[double](max_size)

    with nogil:
        # Process each tree
        for tree_idx in range(num_trees):
            tree_ptr = &filtered_trees[tree_idx]
            
            # Reset per-tree accumulators
            tree_sum.assign(max_size, 0.0)
            tree_squared_sum.assign(max_size, 0.0)
            tree_count.assign(max_size, 0.0)
            
            # Process each rule in the current tree
            for rule_idx in range(tree_ptr.size()):
                rule_ptr = &(tree_ptr[0][rule_idx])
                rule_value = rule_ptr.value
                rule_count = rule_ptr.count
                squared_value = rule_value * rule_value

                # Find valid indices for each column
                skip_rule = False
                for k in range(num_cols):
                    valid_indices_per_col[k].clear()
                    split_size = split_points_list[k].size()
                    for i in range(split_size):
                        if check_value(rule_ptr, columns[k], split_points_list[k][i]):
                            valid_indices_per_col[k].push_back(i)
                    if valid_indices_per_col[k].empty():
                        skip_rule = True
                        break
                        
                if skip_rule:
                    continue

                # Process all valid combinations using multi_index
                multi_index.assign(num_cols, 0)
                done = False
                while not done:
                    # Compute flat index
                    idx = 0
                    for k in range(num_cols):
                        idx *= sizes[k]
                        idx += valid_indices_per_col[k][multi_index[k]]
    
                    # Update statistics
                    tree_sum[idx] += rule_value * rule_count
                    tree_squared_sum[idx] += squared_value * rule_count
                    tree_count[idx] += rule_count

                    # Update multi_index
                    for k in reversed(range(num_cols)):
                        multi_index[k] += 1
                        if multi_index[k] >= valid_indices_per_col[k].size():
                            multi_index[k] = 0
                            if k == 0:
                                done = True  # All combinations have been processed
                        else:
                            break

            # Calculate per-tree statistics
            for idx in range(max_size):
                count = tree_count[idx]

                if count > 0:
                    total_tree_counts[idx] += 1.0
                    average_counts[idx] += count
                    mean = tree_sum[idx] / count

                    e_x2 = tree_squared_sum[idx] / count
                    variance = e_x2 - mean * mean
                    if variance < 0.0:
                        variance = 0.0  # Numerical stability
                    std = sqrt(variance)
                    
                    ensemble_std[idx] += std
                    mean_values[idx] += mean

        # Finalize aggregate statistics
        for idx in range(max_size):
            count = total_tree_counts[idx]
            if count > 0:
                average_counts[idx] /= count

    # Prepare output points
    total_points = max_size
    combined_points = vector[vector[double]](total_points, vector[double](num_cols))
    
    for idx in range(total_points):
        remaining = idx
        for k in reversed(range(num_cols)):
            point_idx = remaining % sizes[k]
            remaining //= sizes[k]
            combined_points[idx][k] = split_points_list[k][point_idx]

    return combined_points, mean_values, ensemble_std, average_counts


@boundscheck(False)
@nonecheck(False)
@wraparound(False)
@initializedcheck(False)
@cdivision(True)
@overflowcheck(False)
@infer_types(True)
cdef double _expected_value(int col, const vector[vector[Rule]] trees):
        cdef:
            vector[vector[Rule]] filtered_trees 
            size_t num_trees 

            const Rule* rule_ptr
            const vector[Rule]* tree_ptr
            size_t k, l, tree_size

            double weighted_sum = 0.0
            double tree_sum 
            double tree_count
            double rule_val, n_count
            double ub, lb


        filtered_trees = filter_trees(trees, vector[int](col))
        num_trees = filtered_trees.size()

        with nogil:
            for k in range(num_trees):
                tree_ptr = &filtered_trees[k]
                tree_size = tree_ptr.size()

                tree_sum = 0.0
                tree_count = 0.0
                
                for l in range(tree_size):
                    rule_ptr = &(tree_ptr[0][l])
                    rule_val = rule_ptr.value
                    n_count = rule_ptr.count
                    
                    tree_sum += rule_val * n_count
                    tree_count += n_count

                if tree_count > 0.0:
                    weighted_sum += (tree_sum / tree_count)

        return weighted_sum
