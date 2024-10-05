from libcpp.vector cimport vector

from cython cimport boundscheck, wraparound, initializedcheck, nonecheck, cdivision, overflowcheck
from .rule cimport filter_trees, get_split_point, check_value
from .utils cimport find_mean, replace_inf, find_min_max
from .lgb cimport analyze_lightgbm

from libc.math cimport INFINITY

import numpy as np
cimport numpy as cnp 
import pandas as pd
   

cdef class Explainer:
    def __init__(self):
        self.trees = vector[vector[Rule]]()
        self.model = None
        self.len_col = -1
        self.columns = None

    def __call__(self, model):
        if "lightgbm" in model.__module__: 
            if "basic" not in model.__module__:
                self.model = model.booster_
            else:
                self.model = model
            self.columns =  self.columns = self.model.feature_name()
            self.len_col = len(self.columns)
            self.trees = analyze_lightgbm(self.model, self.len_col)
        else:
            raise ValueError("MAAAL")

    @boundscheck(False)
    @nonecheck(False)
    @wraparound(False)
    @initializedcheck(False)
    @overflowcheck(False)
    @cdivision(True)
    cpdef object analyze_dependency(self, int main_col, int sub_col):
        """
        Analyzes the dependency between two features by calculating values based on the split points
        of the main and sub columns across all trees.

        Parameters
        ----------
        main_col : int
            The column index of the main feature.
        sub_col : int
            The column index of the sub feature.

        Returns
        -------
        object
            A DataFrame with sub feature split points, main feature split points, and the corresponding values.
        """
        cdef vector[vector[Rule]] filtered_trees = filter_trees(self.trees, main_col, sub_col)
        cdef vector[double] main_split_points = get_split_point(filtered_trees, main_col)
        cdef vector[double] sub_split_points = get_split_point(filtered_trees, sub_col)

        cdef vector[vector[double]] all_values
        cdef vector[double] tree_values
        cdef vector[Rule] tree
        cdef vector[double] mean_values
        cdef vector[double] sub_points
        cdef vector[double] main_points
 
        cdef double sub_point, main_point 
        cdef bint result_main, result_sub
        cdef str main_column_name, sub_column_name
        cdef Rule rule

        cdef object df
        
        with nogil:
            for sub_point in sub_split_points:   
                for main_point in main_split_points:
                    all_values.clear()

                    for tree in filtered_trees:
                        tree_values.clear()

                        for rule in tree:
                            result_main = check_value(&rule, main_col, main_point)
                            result_sub = check_value(&rule, sub_col, sub_point)
                            
                            if result_main & result_sub:
                                tree_values.push_back(rule.value)

                        if tree_values.size() > 0:
                            all_values.push_back(tree_values)

                    if all_values.size() == 0:
                        continue

                    mean_values.push_back(find_mean(all_values))
                    sub_points.push_back(sub_point)
                    main_points.push_back(main_point)
                    

        cdef cnp.ndarray[cnp.float64_t, ndim=1] mean_values_arr = np.asarray(mean_values, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] sub_points_arr = np.asarray(sub_points, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] main_points_arr = np.asarray(main_points, dtype=np.float64)

        main_column_name = self.columns[main_col]
        sub_column_name = self.columns[sub_col]

        df = pd.DataFrame({
            sub_column_name: sub_points_arr,
            main_column_name: main_points_arr,
            'values': mean_values_arr,
        })

        df = df.explode(["values"]).reset_index(drop=True)

        df = replace_inf(df, main_column_name)
        df = replace_inf(df, sub_column_name)

        return df

    @boundscheck(False)
    @initializedcheck(False)
    @nonecheck(False)
    @wraparound(False)
    @overflowcheck(False)
    @cdivision(True)
    cpdef tuple analyze_row(self, object x, bint detailed = True):
        """
        Optimized version of analyze_row function using C++ vectors of doubles.
        
        Parameters and return values remain the same as the original function.
        """
        cdef int[:, ::1] leafs = self.model.predict(x, pred_leaf=True)
        cdef double raw_score = np.mean(self.model.predict(x, raw_score=True))

        cdef int num_rows = leafs.shape[0]
        cdef int num_leafs = leafs.shape[1]
        cdef double split_value

        
        cdef int row, col, i
        cdef size_t j, max_len = 0

        cdef double ub, lb
        cdef int[:] leaf_loc
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] values_1d
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] values_2d
        cdef vector[vector[double]] split_points
        cdef vector[double] col_split_points
        cdef Rule rule
        
        if detailed:
            split_points.resize(self.len_col)
            
            for col in range(self.len_col):
                col_split_points.clear()  # Clear previous values
                split_point_array = np.asarray(get_split_point(self.trees, col), dtype=np.float64)
                
                for i in range(split_point_array.shape[0]):
                    col_split_points.push_back(split_point_array[i])
                
                split_points[col] = col_split_points
                if col_split_points.size() > max_len:
                    max_len = col_split_points.size()
            
            values_2d = np.zeros((self.len_col, max_len), dtype=np.float64)
        else:
            values_1d = np.zeros(self.len_col, dtype=np.float64)
        

        with nogil:
            for row in range(num_rows):
                leaf_loc = leafs[row, :]
                
                for i in range(num_leafs):
                    rule = self.trees[i][leaf_loc[i]]
                    
                    for col in range(self.len_col):
                        ub, lb = rule.ubs[col], rule.lbs[col]
                        
                        if (ub == INFINITY) and (lb == -INFINITY):
                            continue
                        
                        if detailed:
                            col_split_points = split_points[col]
                            for j in range(col_split_points.size()):
                                split_value = col_split_points[j]
                                if lb < split_value <= ub:
                                    values_2d[col, j] += rule.value
                        else:
                            values_1d[col] += rule.value
        
        if detailed:
            for col in range(self.len_col):
                for j in range(max_len):
                    values_2d[col, j] /= num_rows
                    
            split_points_list = []
            for col in range(self.len_col):
                col_split_points = split_points[col]
                np_array = np.zeros(col_split_points.size(), dtype=np.float64)

                for j in range(col_split_points.size()):
                    np_array[j] = col_split_points[j]
                    
                split_points_list.append(np_array)
                
            return values_2d, split_points_list, raw_score
        else:
            for col in range(self.len_col):
                values_1d[col] /= num_rows
                
            return values_1d, raw_score


    @boundscheck(False)
    @nonecheck(False)
    @wraparound(False)
    @initializedcheck(False)
    @overflowcheck(False)
    @cdivision(True)
    cpdef object analyze_feature(self, int col):
        """
        Analyzes a specific feature by calculating the mean, min, and max values
        based on split points across trees for the given column.

        Parameters
        ----------
        col : int
            The column index of the feature to analyze.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the split points (main_point), mean, min, and max values
            for the specified feature.
        """
        cdef vector[vector[Rule]] filtered_trees = filter_trees(self.trees, col)
        cdef vector[double] split_points = get_split_point(filtered_trees, col)
        cdef double point
        cdef vector[vector[double]] all_values
        cdef vector[double] tree_values, points, mean_values, min_vals, max_vals
        cdef vector[Rule] tree
        cdef str column_name
        cdef Rule rule

        with nogil:
            for point in split_points:
                all_values.clear()

                for tree in filtered_trees:
                    tree_values.clear()

                    for rule in tree:
                        if check_value(&rule, col, point):
                            tree_values.push_back(rule.value)

                    if tree_values.size() > 0:
                        all_values.push_back(tree_values)

                if all_values.size() == 0:
                    continue

                min_val, max_val = find_min_max(all_values)
                mean_values.push_back(find_mean(all_values))
                min_vals.push_back(min_val)
                max_vals.push_back(max_val)
                points.push_back(point)
            
        column_name = self.columns[col]


        cdef cnp.ndarray[cnp.float64_t, ndim=1] mean_values_arr = np.asarray(mean_values, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] max_arr = np.asarray(max_vals, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] min_arr = np.asarray(min_vals, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] point_arr = np.asarray(points, dtype=np.float64)


        df = pd.DataFrame({
            column_name: point_arr,
            'mean': mean_values_arr,
            'min': min_arr,
            'max': max_arr,
        })

        df = replace_inf(df, column_name)

        return df
