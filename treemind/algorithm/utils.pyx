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
            vector[vector[Rule]] filtered_trees = filter_trees(trees, col)
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
            double tree_max_val, tree_min_val, count, rule_val, n_count, squared_val
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
                        ensembe_count += (count / iter_count)
                        ensemble_sum += (tree_sum / count)

                    if iter_count > 1:
                        tree_std = tree_std / count  # E[X^2]
                        tree_sum = tree_sum / count  # E[X]
                        tree_std = sqrt(tree_std - tree_sum * tree_sum)

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
           vector[double]] _analyze_interaction(const vector[vector[Rule]] trees,
                                                         int main_col,
                                                         int sub_col):
    cdef:
        # Prepare filtered trees and split points
        vector[vector[Rule]] filtered_trees = filter_trees(trees, main_col, sub_col)
        vector[double] main_split_points = get_split_point(filtered_trees, main_col)
        vector[double] sub_split_points = get_split_point(filtered_trees, sub_col)

        size_t main_points_size = main_split_points.size()
        size_t sub_points_size = sub_split_points.size()
        size_t max_size = main_points_size * sub_points_size
        size_t num_trees = filtered_trees.size()

        cdef vector[double] average_counts = vector[double](max_size, 0.0)
        cdef vector[double] mean_values = vector[double](max_size, 0.0)
        cdef vector[double] total_tree_counts = vector[double](max_size, 0.0)

        cdef vector[double] tree_count = vector[double](max_size, 0.0)
        cdef vector[double] tree_sum = vector[double](max_size, 0.0)
        cdef vector[double] tree_iter = vector[double](max_size, 0.0)

        vector[double] main_points
        vector[double] sub_points

        cdef int index

        # Variables for looping
        size_t point_idx, rule_idx, tree_size, i, j
        const Rule* rule_ptr
        const vector[Rule]* tree_ptr

        double rule_value, rule_count
        double main_point, sub_point

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
            tree_count.assign(max_size, 0.0)
            tree_iter.assign(max_size, 0.0)

            # Loop over rules in the current tree
            for rule_idx in range(tree_size):
                rule_ptr = &(tree_ptr[0][rule_idx])
                rule_value = rule_ptr.value
                rule_count = rule_ptr.count

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
                        tree_count[index] += rule_count
                        tree_iter[index] += 1.0

                
            for point_idx in range(max_size):
                count = tree_count[point_idx]

                if count > 0:
                    mean_values[point_idx] += tree_sum[point_idx] / count
                    average_counts[point_idx] += count / tree_iter[point_idx]
                    total_tree_counts[point_idx] += 1.0

        
        for point_idx in range(max_size):
            count = total_tree_counts[point_idx]
            if count > 0:
                average_counts[point_idx] /= count

        for sub_point in sub_split_points:
            for main_point in main_split_points:
                main_points.push_back(main_point)
                sub_points.push_back(sub_point)

    return main_points, sub_points, mean_values, average_counts


@boundscheck(False)
@nonecheck(False)
@wraparound(False)
@initializedcheck(False)
@cdivision(True)
@overflowcheck(False)
@infer_types(True)
cdef double _expected_value(int col, const vector[vector[Rule]] trees, int [:,:] col_values):
        cdef:
            vector[vector[Rule]] filtered_trees 
            size_t num_trees 

            const Rule* rule_ptr
            const vector[Rule]* tree_ptr
            size_t k, l, tree_size

            double weighted_sum
            double tree_sum 
            double tree_count
            double rule_val, n_count
            double ub, lb
            int[:] row_leafs

            int n_rows = col_values.shape[0]
            int n_leafs 
            int i, m

            bint ub_check, lb_check

            double [:] weighted_sum_rows

        if n_rows > 0:
            n_leafs = col_values.shape[1]
            
            weighted_sum_rows = np.zeros(n_rows, dtype=np.float64)

            with nogil:
                for i in range(n_rows):
                    row_leafs = col_values[i, :]

                    ensemble_sum = 0.0
                    
                    for m in range(n_leafs):
                        rule_ptr = &trees[m][row_leafs[m]]
                        rule_val = rule_ptr.value
                        ub, lb = rule_ptr.ubs[col], rule_ptr.lbs[col]

                        ub_check = (ub == INFINITY)
                        lb_check = (lb == -INFINITY)

                        if ub_check & lb_check:
                            continue

                        ensemble_sum += rule_val
                    
                    weighted_sum_rows[i] = ensemble_sum
                
            return np.mean(np.asarray(weighted_sum_rows))

        else:
            filtered_trees = filter_trees(trees, col)
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

                    weighted_sum += (tree_sum / tree_count)

            return weighted_sum



