from libcpp.vector cimport vector
from .rule cimport filter_trees, get_split_point, check_value

from libc.math cimport INFINITY

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
cdef vector[double] pre_allocate_vector(size_t size) noexcept nogil:
    cdef vector[double] vec
    vec.reserve(size)
    return vec

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



cdef tuple[vector[double], vector[double], vector[double], vector[double], vector[double]] _analyze_feature(int col, const vector[vector[Rule]] trees):
        cdef:
            vector[vector[Rule]] filtered_trees = filter_trees(trees, col)
            vector[double] split_points = get_split_point(filtered_trees, col)
            
            size_t num_splits = split_points.size()
            size_t num_trees = filtered_trees.size()

            vector[double] max_vals, min_vals, mean_values, points, average_counts

            const Rule* rule_ptr
            const vector[Rule]* tree_ptr
            size_t i, k, l, tree_size

            double point, ensemble_sum, ensemble_max_val, ensemble_min_val
            double tree_max_val, tree_min_val, count, rule_val, n_count
            double iter_count, ensemble_count, tree_count

            bint is_valid_rule

        max_vals.reserve(num_splits)
        min_vals.reserve(num_splits)
        mean_values.reserve(num_splits)
        average_counts.reserve(num_splits)
        points.reserve(num_splits)

        with nogil:
            for i in range(num_splits):
                point = split_points[i]

                ensemble_sum = 0.0
                ensemble_max_val = 0.0
                ensemble_min_val = 0.0
                ensembe_count = 0.0
                tree_count = 0.0
                
                for k in range(num_trees):
                    tree_ptr = &filtered_trees[k]
                    tree_size = tree_ptr.size()

                    tree_sum = 0.0
                    count = 0.0
                    iter_count = 0.0
                    tree_max_val = -INFINITY
                    tree_min_val = INFINITY
                    
                    for l in range(tree_size):
                        rule_ptr = &(tree_ptr[0][l])
                        is_valid_rule = check_value(rule_ptr, col, point)
                        
                        if is_valid_rule:
                            rule_val = rule_ptr.value
                            n_count = rule_ptr.count

                            tree_sum += rule_val * n_count
                            count += n_count
                            iter_count += 1
                            
                            tree_max_val = cmax(tree_max_val, rule_val)
                            tree_min_val = cmin(tree_min_val, rule_val)                            
                    
                    if count > 0:
                        tree_count += 1.0
                        ensembe_count += (count / iter_count)

                        ensemble_sum += (tree_sum / count)
                        ensemble_max_val += tree_max_val
                        ensemble_min_val += tree_min_val
                
                if ensemble_sum != 0.0:
                    points.push_back(point)
                    mean_values.push_back(ensemble_sum)
                    min_vals.push_back(ensemble_min_val)
                    max_vals.push_back(ensemble_max_val)
                    average_counts.push_back(ensembe_count / tree_count)

        return points, mean_values, min_vals, max_vals, average_counts
        


cdef tuple[vector[double], 
            vector[double], 
            vector[double],
            vector[double]] _analyze_interaction(const vector[vector[Rule]] trees, 
                                                int main_col, 
                                                int sub_col):
    cdef:
        vector[vector[Rule]] filtered_trees = filter_trees(trees, main_col, sub_col)
        vector[double] main_split_points = get_split_point(filtered_trees, main_col)
        vector[double] sub_split_points = get_split_point(filtered_trees, sub_col)

        size_t main_points_size = main_split_points.size()
        size_t sub_points_size = sub_split_points.size()
        size_t max_size = main_points_size * sub_points_size

        vector[double] mean_values, sub_points, main_points, average_counts
                
        double sub_point, main_point
        size_t i, j, k, l, tree_size
        const Rule* rule_ptr
        const vector[Rule]* tree_ptr

        double count, tree_sum, ensemble_sum
        double n_count, iter_count, ensemble_count, tree_count

        size_t num_trees = filtered_trees.size()
        bint rule_valid

    mean_values.reserve(max_size)
    sub_points.reserve(max_size)
    main_points.reserve(max_size)
    average_counts.reserve(max_size)

    with nogil:
        for i in range(sub_points_size):
            sub_point = sub_split_points[i]
            for j in range(main_points_size):
                main_point = main_split_points[j]
                ensemble_sum = 0.0
                ensembe_count = 0.0
                tree_count = 0.0
                
                for k in range(num_trees):
                    tree_ptr = &filtered_trees[k]
                    tree_size = tree_ptr.size()
                    
                    tree_sum = 0.0
                    count = 0.0
                    tree_sum = 0.0
                    iter_count = 0.0
                    
                    for l in range(tree_size):
                        rule_ptr = &(tree_ptr[0][l])
                        rule_valid = check_value(rule_ptr, main_col, main_point) & check_value(rule_ptr, sub_col, sub_point)

                        if rule_valid:
                            n_count = rule_ptr.count
                            tree_sum += rule_ptr.value * n_count
                            count += n_count
                            iter_count += 1.0
                    
                    if count > 0:
                        ensemble_sum += (tree_sum / count)
                        ensembe_count += (count / iter_count)
                        tree_count += 1.0
                
                if ensemble_sum != 0.0:
                    mean_values.push_back(ensemble_sum)
                    sub_points.push_back(sub_point)
                    main_points.push_back(main_point)
                    average_counts.push_back(ensembe_count / tree_count)


    return main_points, sub_points, mean_values, average_counts


