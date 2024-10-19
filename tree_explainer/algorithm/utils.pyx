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



cdef tuple[vector[double], vector[double], vector[double], vector[double]] _analyze_feature(int col, vector[vector[Rule]] trees):
        cdef:
            vector[vector[Rule]] filtered_trees = filter_trees(trees, col)
            vector[double] split_points = get_split_point(filtered_trees, col)
            
            size_t estimated_size = split_points.size()

            vector[double] max_vals = pre_allocate_vector(estimated_size)
            vector[double] min_vals = pre_allocate_vector(estimated_size)
            vector[double] mean_values = pre_allocate_vector(estimated_size)
            vector[double] points = pre_allocate_vector(estimated_size)

            Rule* rule_ptr
            vector[Rule]* tree_ptr
            size_t i, k, l

            double point, ensemble_sum, ensemble_max_val, ensemble_min_val, tree_max_val, tree_min_val, count, rule_val
    
        with nogil:
            for i in range(split_points.size()):
                point = split_points[i]

                ensemble_sum = 0.0
                ensemble_max_val = 0.0
                ensemble_min_val = 0.0
                
                for k in range(filtered_trees.size()):
                    tree_ptr = &filtered_trees[k]
                    tree_sum = 0.0
                    count = 0.0
                    tree_max_val = -INFINITY
                    tree_min_val = INFINITY
                    
                    for l in range(tree_ptr.size()):
                        rule_ptr = &(tree_ptr[0][l])
                        if check_value(rule_ptr, col, point):
                            rule_val = rule_ptr.value
                            tree_sum += rule_val
                            count += 1.0

                            tree_max_val = cmax(tree_max_val, rule_val)
                            tree_min_val = cmin(tree_min_val, rule_val)                            
                    
                    if count > 0:
                        ensemble_sum += (tree_sum / count)
                        ensemble_max_val += tree_max_val
                        ensemble_min_val += tree_min_val
                
                
                if ensemble_sum == 0.0:
                    continue
                    
                points.push_back(point)
                mean_values.push_back(ensemble_sum)
                min_vals.push_back(ensemble_min_val)
                max_vals.push_back(ensemble_max_val)

        return points, mean_values, min_vals, max_vals
        


cdef tuple[vector[double], 
            vector[double], 
            vector[double]] _analyze_dependency(vector[vector[Rule]] trees, 
                                                int main_col, 
                                                int sub_col):
    cdef:
        vector[vector[Rule]] filtered_trees = filter_trees(trees, main_col, sub_col)
        vector[double] main_split_points = get_split_point(filtered_trees, main_col)
        vector[double] sub_split_points = get_split_point(filtered_trees, sub_col)

        size_t estimated_size = main_split_points.size() * sub_split_points.size()

        vector[double] mean_values = pre_allocate_vector(estimated_size)
        vector[double] sub_points = pre_allocate_vector(estimated_size)
        vector[double] main_points = pre_allocate_vector(estimated_size)

        double sub_point, main_point
        size_t i, j, k, l
        Rule* rule_ptr
        vector[Rule]* tree_ptr

        double count, tree_sum, ensemble_sum

    with nogil:
        for i in range(sub_split_points.size()):
            sub_point = sub_split_points[i]
            for j in range(main_split_points.size()):
                main_point = main_split_points[j]
                ensemble_sum = 0.0
                
                for k in range(filtered_trees.size()):
                    tree_ptr = &filtered_trees[k]
                    tree_sum = 0.0
                    count = 0.0
                    
                    for l in range(tree_ptr.size()):
                        rule_ptr = &(tree_ptr[0][l])
                        if check_value(rule_ptr, main_col, main_point) & check_value(rule_ptr, sub_col, sub_point):
                            tree_sum += rule_ptr.value
                            count += 1.0
                    
                    if count > 0:
                        ensemble_sum += (tree_sum / count)
                
                if ensemble_sum == 0.0:
                    continue
                    
                mean_values.push_back(ensemble_sum)
                sub_points.push_back(sub_point)
                main_points.push_back(main_point)


    return main_points, sub_points, mean_values