from libcpp.vector cimport vector

from libc.math cimport INFINITY

import numpy as np
cimport numpy as cnp 
import pandas as pd

from libcpp.pair cimport pair

from cython cimport boundscheck, wraparound, initializedcheck, nonecheck, cdivision, overflowcheck, infer_types

from .rule cimport get_split_point
from .utils cimport _analyze_feature, _analyze_dependency, add_lower_bound
from .lgb cimport analyze_lightgbm
from .xgb cimport analyze_xgboost, xgb_leaf_correction, convert_d_matrix

from collections import Counter
from itertools import combinations

cdef vector[pair[double, double]] feature_ranges

cdef class Explainer:
    def __init__(self):
        self.trees = vector[vector[Rule]]()
        self.model = None
        self.len_col = -1
        self.columns = None
        self.model_type = "None"


    def __call__(self, model):
        cdef str module_name = model.__module__

        if "lightgbm" in module_name: 
            if "basic" not in module_name:
                self.model = model.booster_
            else:
                self.model = model
            self.columns = self.model.feature_name()
            self.len_col = len(self.columns)
            self.model_type = "lightgbm"
            self.trees = analyze_lightgbm(self.model, self.len_col)

        elif "xgboost" in module_name:
            if "core" not in module_name:
                self.model = model.get_booster()
            else:
                self.model = model

            self.len_col = self.model.num_features()

            if self.model.feature_names is None:
                self.columns = [f"Feature_{i}" for i in range(self.len_col)]
            else:
                self.columns = self.model.feature_name()

            self.model_type = "xgboost"
            self.trees = analyze_xgboost(self.model, self.len_col)
    
        else:
            raise ValueError("MAAAL")

    @boundscheck(False)
    @nonecheck(False)
    @wraparound(False)
    @initializedcheck(False)
    @overflowcheck(False)
    @cdivision(True)
    @infer_types(True)
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
        if self.len_col == -1:
            raise ValueError("Explainer(model) must be called before this operation.")

        if main_col >= self.len_col or sub_col >= self.len_col:
            raise ValueError("'main_col' and 'sub_col' cannot be greater than or equal to the total number of columns used to train the model.")

        if main_col < 0 or sub_col < 0:
            raise ValueError("'main_col' and 'sub_col' cannot be negative.")

        if main_col == sub_col:
            raise ValueError("'main_col' and 'sub_col' cannot be the same.")

        cdef:
             double sub_point, main_point
             str main_column_name, sub_column_name
             object df
             vector[double] mean_values, sub_points,main_points
                
        main_points, sub_points, mean_values = _analyze_dependency(self.trees, main_col, sub_col)
                            
        main_column_name = self.columns[main_col]
        sub_column_name = self.columns[sub_col]

        df = pd.DataFrame({
            f"{main_column_name}_ub": main_points,
            f"{sub_column_name}_ub": sub_points,
            'value': mean_values,
        })

        df = df.explode(["value"]).reset_index(drop=True)

        add_lower_bound(df, 0, main_column_name)
        add_lower_bound(df, 2, sub_column_name)
       
        return df    

    @boundscheck(False)
    @nonecheck(False)
    @wraparound(False)
    @initializedcheck(False)
    @overflowcheck(False)
    @cdivision(True)
    @infer_types(True)
    cpdef tuple analyze_row(self, object x, bint detailed = True):
        """
        Optimized version of analyze_row function using C++ vectors of doubles.
        
        Parameters and return values remain the same as the original function.
        """
        if self.len_col == -1:
            raise ValueError("Explainer(model) must be called before this operation.")

        if self.model_type == "xgboost":
            x = convert_d_matrix(x)

        cdef:
            int[:, ::1] leafs = self.model.predict(x, pred_leaf=True).astype(np.int32)
            double raw_score = np.mean(self.model.predict(x, raw_score=True) if self.model_type == "lightgbm"  else self.model.predict(x, output_margin=True).astype(np.float64)).item()
        
            int num_rows = leafs.shape[0]
            int num_leafs = leafs.shape[1]

            int row, col, size, i
            size_t j, max_len = 0
            double ub, lb, split_value
            int[:] leaf_loc
            cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] values_1d, np_array
            cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] values_2d
            vector[vector[double]] split_points
            vector[double] col_split_points
            Rule rule

            list split_points_list

        if self.model_type == "xgboost":
            leafs = xgb_leaf_correction(self.trees, leafs)

        if detailed:
            split_points.resize(self.len_col)
            
            for col in range(self.len_col):
                col_split_points = get_split_point(self.trees, col)
                
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
            values_2d[col, j] /= num_rows
            split_points_list = [np.array(split_points[col]) for col in range(self.len_col)]

            return values_2d, split_points_list, raw_score
        else:
            values_1d /= num_rows
            
            return values_1d, raw_score

    @boundscheck(False)
    @nonecheck(False)
    @wraparound(False)
    @initializedcheck(False)
    @overflowcheck(False)
    @cdivision(True)
    @infer_types(True)
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
        if self.len_col == -1:
            raise ValueError("Explainer(model) must be called before this operation.")

        if col >= self.len_col:
            raise ValueError("'col' cannot be greater than or equal to the total number of columns used to train the model.")

        if col < 0:
            raise ValueError("'col' cannot be negative.")

        cdef:
            vector[double] points, mean_values, min_vals, max_vals
            str column_name
            object df

        points, mean_values, min_vals, max_vals = _analyze_feature(col, self.trees)
            
        column_name = self.columns[col]

        df = pd.DataFrame({
            f'{column_name}_ub': points,
            'mean': mean_values,
            'min': min_vals,
            'max': max_vals,
        })

        df.insert(0, f"{column_name}_lb", df[f'{column_name}_ub'].shift(1).fillna(-np.inf))

        return df

    @boundscheck(False)
    @nonecheck(False)
    @wraparound(False)
    @initializedcheck(False)
    @overflowcheck(False)
    @cdivision(True)
    @infer_types(True)
    cpdef object count_node(self, bint interaction=True):
        cdef:
            vector[Rule] rule_set
            Rule rule
            object df, combination_counts = Counter()
            list finite_indices, data, columns
            int i, n, comb_size = 2 if interaction else 1
            tuple comb

        for rule_set in self.trees:
            for rule in rule_set:
                finite_indices = [
                    i
                    for i in range(self.len_col)
                    if rule.lbs[i] != -INFINITY or rule.ubs[i] != INFINITY
                ]
                n = len(finite_indices)

                if n >= comb_size:
                    for comb in combinations(finite_indices, comb_size):
                        combination_counts[comb] += 1

        if interaction:
            data = [(comb[0], comb[1], count) for comb, count in combination_counts.items()]
            columns = ["column1_index", "column2_index", "count"]
        else:
            data = [(comb[0], count) for comb, count in combination_counts.items()]
            columns = ["column_index", "count"]

        df = pd.DataFrame(data, columns=columns)

        return df.sort_values("count", ascending=False).reset_index(drop=True)
