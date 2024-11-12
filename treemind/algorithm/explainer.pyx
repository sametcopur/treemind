from libcpp.vector cimport vector

from libc.math cimport INFINITY

import numpy as np
cimport numpy as cnp 
import pandas as pd

from libcpp.pair cimport pair

from cython cimport boundscheck, wraparound, initializedcheck, nonecheck, cdivision, overflowcheck, infer_types

from .rule cimport update_leaf_counts
from .utils cimport _analyze_feature, _analyze_interaction, add_lower_bound, _expected_value, _analyze_multi_interaction
from .lgb cimport analyze_lightgbm
from .xgb cimport analyze_xgboost, convert_d_matrix, xgb_leaf_correction
from .cb cimport analyze_catboost

from collections import Counter
from itertools import combinations

cdef vector[pair[double, double]] feature_ranges

cdef class Explainer:
    def __init__(self):
        self.trees = vector[vector[Rule]]()
        self.model = None
        self.len_col = -1
        self.columns = None
        self.model_type = "none"

    def __repr__(self) -> str:
        return f"Explainer(model={self.model_type})"

    def __call__(self, model):
        if not hasattr(model, '__module__'):
            raise ValueError("The provided model is neither a LightGBM nor an XGBoost model. Please provide a supported model type.")

        module_name = model.__module__

        # Handle LightGBM models
        if "lightgbm" in module_name: 
            if "basic" not in module_name:
                self.model = model.booster_
            else:
                self.model = model
            self.columns = self.model.feature_name()
            self.len_col = len(self.columns)
            self.model_type = "lightgbm"
            self.trees = analyze_lightgbm(self.model, self.len_col)

        # Handle XGBoost models
        elif "xgboost" in module_name:
            if "core" not in module_name:
                self.model = model.get_booster()
            else:
                self.model = model

            self.len_col = self.model.num_features()

            if self.model.feature_names is None:
                self.columns = [f"Feature_{i}" for i in range(self.len_col)]
            else:
                self.columns = self.model.feature_names

            self.model_type = "xgboost"
            self.trees = analyze_xgboost(self.model, self.len_col)

        elif "catboost" in module_name:
            self.model = model
            self.len_col = self.model.n_features_in_
            self.columns = model.feature_names_
            self.trees = analyze_catboost(self.model, self.len_col)

        # Raise an error if the model is not LightGBM or XGBoost
        else:
            raise ValueError("The provided model is neither a LightGBM nor an XGBoost model. Please provide a supported model type.")


    @boundscheck(False)
    @nonecheck(False)
    @wraparound(False)
    @initializedcheck(False)
    @overflowcheck(False)
    @cdivision(True)
    @infer_types(True)
    cpdef object analyze_interaction(self, int main_col, int sub_col, object back_data = None):
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
             vector[double] mean_values, sub_points,main_points, counts, stds
             vector[vector[Rule]] trees = self.trees

            
        if back_data is not None:
            trees = update_leaf_counts(trees, self.model, back_data, self.model_type)
 
        main_points, sub_points, mean_values, stds, counts = _analyze_interaction(trees, main_col, sub_col)

        main_column_name = self.columns[main_col]
        sub_column_name = self.columns[sub_col]

        df = pd.DataFrame({
            f"{main_column_name}_ub": main_points,
            f"{sub_column_name}_ub": sub_points,
            'value': mean_values,
            'std': stds,
            'count': counts
        })

        if df.shape[0] == 0:
            raise ValueError(f"No interaction found between feature {main_col} and {sub_col}.")

        df = df.explode(["value"]).reset_index(drop=True)
        df.loc[:, "value"] -= (df["value"] * df["count"]).sum() / df["count"].sum()

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
    cpdef cnp.ndarray[cnp.float64_t, ndim=2] analyze_data(self, object x, object back_data = None):
        if self.len_col == -1:
            raise ValueError("Explainer(model) must be called before this operation.")

        cdef:
            double  ensemble_sum 
            size_t i, j, num_trees, tree_size
            double rule_val, n_count, point, expected_value
            vector[vector[Rule]] trees = self.trees

            int[:,:] x_ 
            int[:] tree_loc 

            int row, col, num_rows 

            vector[double] expected_values

            const Rule* rule_ptr
            const vector[Rule]* tree_ptr
        
            double[:,:] values 
            object x_dmatrix 

        if self.model_type == "xgboost":
            x_dmatrix = convert_d_matrix(x)
            x_ = self.model.predict(x_dmatrix, pred_leaf=True).astype(np.int32)
            x_ = xgb_leaf_correction(trees, x_)

        else:
            x_ = self.model.predict(x, pred_leaf=True).astype(np.int32)

        num_rows = x_.shape[0]
        values = np.empty((num_rows, self.len_col), dtype=np.float64)

        if back_data is not None:
            trees = update_leaf_counts(trees, self.model, back_data, self.model_type)

        num_trees = trees.size()



        expected_values.resize(self.len_col)
        for col in range(self.len_col):
            expected_values[col] = _expected_value(col, trees)

        with nogil:
            for col in range(self.len_col):
                expected_value = expected_values[col]
                
                for row in range(num_rows):
                    tree_loc = x_[row, :]
                    ensemble_sum = 0.0

                    for i in range(num_trees):
                        tree_ptr = &trees[i]
                        rule_ptr = &(tree_ptr[0][tree_loc[i]])

                        if not ((rule_ptr.lbs[col] == -INFINITY )and (rule_ptr.ubs[col] == INFINITY)):
                            ensemble_sum += rule_ptr.value

                    values[row, col] = ensemble_sum - expected_value

        return np.asarray(values)

    @boundscheck(False)
    @nonecheck(False)
    @wraparound(False)
    @initializedcheck(False)
    @overflowcheck(False)
    @cdivision(True)
    @infer_types(True)
    cpdef object analyze_feature(self, int col, object back_data = None):
        if not isinstance(col, int):
            raise ValueError("The 'col' parameter must be an integer.")

        if self.len_col == -1:
            raise ValueError("Explainer(model) must be called before this operation.")

        if col >= self.len_col:
            raise ValueError("'col' cannot be greater than or equal to the total number of columns used to train the model.")

        if col < 0:
            raise ValueError("'col' cannot be negative.")

        cdef:
            double[:] points, mean_values, stds, counts
            vector[vector[Rule]] trees = self.trees
            str column_name
            object df

        if back_data is not None:
            trees = update_leaf_counts(trees, self.model, back_data, self.model_type)

        points, mean_values, stds, counts = _analyze_feature(col, trees)
                    
        column_name = self.columns[col]

        df = pd.DataFrame({
            f'{column_name}_ub': np.asarray(points),
            'mean': np.asarray(mean_values),
            'std': np.asarray(stds),
            "count": np.asarray(counts)
        })
        df.loc[:, "mean"] -= (df["mean"] * df["count"]).sum() / df["count"].sum()
        df.insert(0, f"{column_name}_lb", df[f'{column_name}_ub'].shift(1).fillna(-np.inf))

        return df

    @boundscheck(False)
    @nonecheck(False)
    @wraparound(False)
    @initializedcheck(False)
    @overflowcheck(False)
    @cdivision(True)
    @infer_types(True)
    cpdef object count_node(self, int order=2):
        if self.len_col == -1:
            raise ValueError("Explainer(model) must be called before this operation.")

        if order >= self.len_col:
            raise ValueError("'order' cannot be greater than or equal to the total number of columns used to train the model.")

        if order < 1:
            raise ValueError("The 'order' parameter must be a positive integer.")


        cdef:
            vector[Rule] rule_set
            Rule rule
            object df, combination_counts = Counter()
            list finite_indices, data, columns
            int i, n
            tuple comb

        for rule_set in self.trees:
            for rule in rule_set:
                finite_indices = [
                    i
                    for i in range(self.len_col)
                    if rule.lbs[i] != -INFINITY or rule.ubs[i] != INFINITY
                ]
                n = len(finite_indices)

                if n >= order:
                    for comb in combinations(finite_indices, order):
                        combination_counts[comb] += 1

        # Create column names dynamically based on order
        columns = [f"column{i+1}_index" for i in range(order)]
        columns.append("count")

        # Prepare data for DataFrame
        data = [(*(comb), count) for comb, count in combination_counts.items()]

        df = pd.DataFrame(data, columns=columns)
        return df.sort_values("count", ascending=False).reset_index(drop=True)
            
    @boundscheck(False)
    @nonecheck(False)
    @wraparound(False)
    @initializedcheck(False)
    @overflowcheck(False)
    @cdivision(True)
    @infer_types(True)
    cpdef object analyze_multi_interaction(self, list columns, object back_data = None):
        if self.len_col == -1:
            raise ValueError("Explainer(model) must be called before this operation.")
        
        # Convert columns list to vector[int]
        cdef:
            vector[int] col_indices
            int col
            vector[vector[Rule]] trees = self.trees
            vector[vector[double]] points
            vector[double] mean_values, ensemble_std, counts
            size_t i, col_idx
            cdef size_t num_cols = col_indices.size()

        col_indices.reserve(len(columns))
        
        for col in columns:
            col_indices.push_back(col)

        num_cols = col_indices.size()
        
        # Validate input columns
        
        if num_cols < 1:
            raise ValueError("At least one columns must be provided for interaction analysis.")

        elif num_cols > self.len_col:
            raise ValueError("The length of columns must be smaller than the total number of columns used to train the model.")
    
        for col in col_indices:
            if col >= self.len_col:
                raise ValueError(f"Column index {col} cannot be greater than or equal to the total number of columns used to train the model.")
            if col < 0:
                raise ValueError("Column indices cannot be negative.")
                
        # Check for duplicate columns
        seen = set()
        for col in columns:
            if col in seen:
                raise ValueError("Duplicate column indices are not allowed.")
            seen.add(col)

            
        if back_data is not None:
            trees = update_leaf_counts(trees, self.model, back_data, self.model_type)
        
        # Analyze interactions
        points, mean_values, ensemble_std, counts = _analyze_multi_interaction(trees, col_indices)
        
        # Get column names using regular Python list
        column_names = [self.columns[idx] for idx in columns]
        
        # Create DataFrame
        df_dict = {}
        
        # Add points columns
        for i in range(num_cols):
            df_dict[f"{column_names[i]}_ub"] = [row[i] for row in points]
        
        # Add statistics columns
        df_dict.update({
            'value': mean_values,
            'std': ensemble_std,
            'count': counts
        })
        
        df = pd.DataFrame(df_dict)
        
        if df.shape[0] == 0:
            raise ValueError(f"No interaction found between the specified features: {column_names}")
        
        # Center the values
        df.loc[:, "value"] -= (df["value"] * df["count"]).sum() / df["count"].sum()
        
        # Add lower bounds for all columns
        for i, col_name in enumerate(column_names):
            add_lower_bound(df, i * 2, col_name)
        
        return df
