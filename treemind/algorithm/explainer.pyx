from libcpp.vector cimport vector

from libc.math cimport INFINITY

import pandas as pd

from libcpp.pair cimport pair

from cython cimport boundscheck, wraparound, initializedcheck, nonecheck, cdivision, overflowcheck, infer_types

from .rule cimport update_leaf_counts
from .utils cimport add_lower_bound, _analyze_feature
from .lgb cimport analyze_lightgbm
from .xgb cimport analyze_xgboost
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
            raise ValueError("The provided model isn't a lightgbm, xgboost or catboost model. Please provide a supported model type.")

        module_name = model.__module__

        # Handle lightgbm models
        if "lightgbm" in module_name: 
            if "basic" not in module_name:
                self.model = model.booster_
            else:
                self.model = model

            if self.model._Booster__num_class != 1:
                raise ValueError("Multiclass lightgbm models are not supported yet.")
            
            self.columns = self.model.feature_name()
            self.len_col = len(self.columns)
            self.model_type = "lightgbm"
            self.trees = analyze_lightgbm(self.model, self.len_col)

        # Handle xgboost models
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
            self.model_type = "catboost"

        # Raise an error if the model is not lightgbm or xgboost
        else:
            raise ValueError("The provided model isn't a lightgbm, xgboost or catboost model. Please provide a supported model type.")

    @boundscheck(False)
    @nonecheck(False)
    @wraparound(False)
    @initializedcheck(False)
    @overflowcheck(False)
    @cdivision(True)
    @infer_types(True)
    cpdef object analyze_feature(self, object columns, object back_data = None):
        if self.len_col == -1:
            raise ValueError("Explainer(model) must be called before this operation.")
        
        # Convert columns list to vector[int]
        cdef:
            vector[int] col_indices
            int col
            vector[vector[Rule]] trees = self.trees
            vector[vector[double]] points
            vector[double] mean_values, ensemble_std, counts
            int num_cols, i, col_idx

        
        if isinstance(columns, int):
            col_indices.push_back(columns)

        elif isinstance(columns, list):
            for col in columns:
                col_indices.push_back(col)
        else:
            raise ValueError("Invalid type for 'columns'. Expected an int or a list.")

        num_cols = <int>col_indices.size()
        
        # Validate input columns
        if num_cols < 1:
            raise ValueError("At least one columns must be provided for feature analysis.")

        elif num_cols > self.len_col:
            raise ValueError("The length of columns must be smaller than the total number of columns used to train the model.")
    
        for col in col_indices:
            if col >= self.len_col:
                raise ValueError(f"Column index {col} cannot be greater than or equal to the total number of columns used to train the model.")
            if col < 0:
                raise ValueError("Column indices cannot be negative.")
                
        # Check for duplicate columns

        if isinstance(columns, list):
            seen = set()
            for col in columns:
                if col in seen:
                    raise ValueError("Duplicate column indices are not allowed.")
                seen.add(col)

        if back_data is not None:
            trees = update_leaf_counts(trees, self.model, back_data, self.model_type)
        
        # Analyze interactions
        points, mean_values, ensemble_std, counts = _analyze_feature(trees, col_indices)
        
        # Get column names using regular Python list
        column_names = [self.columns[idx] for idx in col_indices]
        
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
            