from libcpp.vector cimport vector
from .rule cimport filter_trees, get_split_point, check_value

from libc.math cimport INFINITY, sqrt

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
cdef tuple _analyze_feature(int col, const vector[vector[Rule]] trees):
        cdef:
            vector[int] column = [col]
            vector[vector[Rule]] filtered_trees = filter_trees(trees, column)
            vector[double] split_points = get_split_point(filtered_trees, col)
            
            size_t num_splits = split_points.size()
            size_t num_trees = filtered_trees.size()

            double[:] stds = np.empty(num_splits, dtype=np.float64)
            double[:] mean_values = np.empty(num_splits, dtype=np.float64)
            double[:] average_counts = np.empty(num_splits, dtype=np.float64)
            double[:] points = np.empty(num_splits, dtype=np.float64)

            const Rule* rule_ptr
            const vector[Rule]* tree_ptr
            size_t i, k, l, tree_size

            double point, ensemble_sum, ensemble_std
            double tree_max_val, tree_min_val, count, rule_val, n_count, squared_val, tree_exp
            double iter_count, ensemble_count, tree_count

            bint is_valid_rule

        with nogil:
            for i in range(num_splits):
                point = split_points[i]

                ensemble_sum = 0.0
                ensemble_std = 0.0
                ensembe_count = 0.0
                tree_count = 0.0
                
                for k in range(num_trees):
                    tree_ptr = &filtered_trees[k]
                    tree_size = tree_ptr.size()

                    tree_sum = 0.0
                    tree_std = 0.0
                    count = 0.0
                    iter_count = 0.0
                    
                    for l in range(tree_size):
                        rule_ptr = &(tree_ptr[0][l])
                        is_valid_rule = check_value(rule_ptr, col, point)
                        
                        if is_valid_rule:
                            rule_val = rule_ptr.value
                            n_count = rule_ptr.count
                            squared_val = rule_val * rule_val

                            tree_sum += rule_val * n_count
                            tree_std += squared_val * n_count  # E[X^2]
                            count += n_count
                            iter_count += 1
                     
                    
                    if count > 0:
                        tree_count += 1.0
                        tree_exp = tree_sum / count

                        ensembe_count += count 
                        ensemble_sum += tree_exp

                    if iter_count > 1:
                        tree_std = tree_std / count  # E[X^2]

                        variance = tree_std - tree_exp * tree_exp
                        if variance < 0.0:
                            variance = 0.0  # Numerical stability
                        tree_std = sqrt(variance)
                        ensemble_std += tree_std

                
                if ensemble_sum != 0.0:
                    points[i] = point
                    mean_values[i] = ensemble_sum
                    stds[i] = ensemble_std
                    average_counts[i] = (ensembe_count / tree_count)

        return points, mean_values, stds, average_counts
        
@boundscheck(False)
@nonecheck(False)
@wraparound(False)
@initializedcheck(False)
@cdivision(True)
@overflowcheck(False)
@infer_types(True)
cdef tuple[vector[double],
           vector[double],
           vector[double],
           vector[double],
           vector[double]] _analyze_interaction(const vector[vector[Rule]] trees,
                                                         int main_col,
                                                         int sub_col):
    cdef:
        # Prepare filtered trees and split points
        vector[int] columns = [main_col, sub_col]
        vector[vector[Rule]] filtered_trees = filter_trees(trees, columns)
        vector[double] main_split_points = get_split_point(filtered_trees, main_col)
        vector[double] sub_split_points = get_split_point(filtered_trees, sub_col)

        size_t main_points_size = main_split_points.size()
        size_t sub_points_size = sub_split_points.size()
        size_t max_size = main_points_size * sub_points_size
        size_t num_trees = filtered_trees.size()

        cdef vector[double] average_counts = vector[double](max_size, 0.0)
        cdef vector[double] mean_values = vector[double](max_size, 0.0)
        cdef vector[double] total_tree_counts = vector[double](max_size, 0.0)
        cdef vector[double] ensemble_std = vector[double](max_size, 0.0)

        cdef vector[double] tree_count = vector[double](max_size, 0.0)
        cdef vector[double] tree_sum = vector[double](max_size, 0.0)
        cdef vector[double] tree_squared_sum = vector[double](max_size, 0.0)
        cdef vector[double] tree_iter = vector[double](max_size, 0.0)

        vector[double] main_points
        vector[double] sub_points

        cdef int index

        # Variables for looping
        size_t point_idx, rule_idx, tree_size, i, j
        const Rule* rule_ptr
        const vector[Rule]* tree_ptr

        double rule_value, rule_count, squared_value
        double main_point, sub_point
        double e_x2, std, variance

        cdef vector[int] valid_main_points
        cdef vector[int] valid_sub_points

    valid_main_points.reserve(main_points_size)
    valid_sub_points.reserve(sub_points_size)
    main_points.reserve(max_size)
    sub_points.reserve(max_size)

    with nogil:
        # Loop over trees
        for tree_idx in range(num_trees):
            tree_ptr = &filtered_trees[tree_idx]
            tree_size = tree_ptr.size()

            tree_sum.assign(max_size, 0.0)
            tree_squared_sum.assign(max_size, 0.0)
            tree_count.assign(max_size, 0.0)
            tree_iter.assign(max_size, 0.0)

            # Loop over rules in the current tree
            for rule_idx in range(tree_size):
                rule_ptr = &(tree_ptr[0][rule_idx])
                rule_value = rule_ptr.value
                rule_count = rule_ptr.count
                squared_value = rule_value * rule_value

                # Collect valid main points for this rule
                valid_main_points.clear()
                for i in range(main_points_size):
                    main_point = main_split_points[i]
                    if check_value(rule_ptr, main_col, main_point):
                        valid_main_points.push_back(i)

                # Skip if no valid main points
                if valid_main_points.size() == 0:
                    continue

                # Collect valid sub points for this rule
                valid_sub_points.clear()
                for i in range(sub_points_size):
                    sub_point = sub_split_points[i]
                    if check_value(rule_ptr, sub_col, sub_point):
                        valid_sub_points.push_back(i)

                # Skip if no valid sub points
                if valid_sub_points.size() == 0:
                    continue

                for i in valid_sub_points:
                    for j in valid_main_points:
                        index = main_points_size * i + j

                        tree_sum[index] += rule_value * rule_count
                        tree_squared_sum[index] += squared_value * rule_count
                        tree_count[index] += rule_count
                        tree_iter[index] += 1.0

            # Calculate per-tree mean and standard deviation
            for point_idx in range(max_size):
                count = tree_count[point_idx]
                iter_count = tree_iter[point_idx]

                if count > 0:
                    total_tree_counts[point_idx] += 1.0
                    average_counts[point_idx] += count 
                    mean = tree_sum[point_idx] / count

                    if iter_count > 1:
                        e_x2 = tree_squared_sum[point_idx] / count
                        variance = e_x2 - mean * mean
                        if variance < 0.0:
                            variance = 0.0  # Numerical stability
                        std = sqrt(variance)
                        ensemble_std[point_idx] += std

                    mean_values[point_idx] += mean

        # Finalize calculations
        for point_idx in range(max_size):
            count = total_tree_counts[point_idx]
            if count > 0:
                average_counts[point_idx] /= count

        # Prepare points for output
        for i in range(sub_points_size):
            sub_point = sub_split_points[i]
            for j in range(main_points_size):
                main_point = main_split_points[j]
                main_points.push_back(main_point)
                sub_points.push_back(sub_point)

    return main_points, sub_points, mean_values, ensemble_std, average_counts

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
           vector[double]] _analyze_multi_interaction(const vector[vector[Rule]]& trees,
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
