from libcpp.vector cimport vector

from libc.math cimport INFINITY

import numpy as np
cimport numpy as cnp 
import pandas as pd

from libcpp.pair cimport pair

from cython cimport boundscheck, wraparound, initializedcheck, nonecheck, cdivision, overflowcheck, infer_types

from .rule cimport filter_trees, check_value
from .utils cimport _analyze_feature, _analyze_interaction, add_lower_bound, _expected_value
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
    cpdef object analyze_interaction(self, int main_col, int sub_col):
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

 
        main_points, sub_points, mean_values, stds, counts = _analyze_interaction(self.trees, main_col, sub_col)

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
            double tree_count, point_col_sum, point_iter_count, point_count, ensemble_sum 
            size_t i, j, num_trees, tree_size
            double rule_val, n_count, point, expected_value

            int[:] leaf_loc
            double[:,:] x_ = np.asarray(x, dtype=np.float64)
            int[:,:] back_data_ 

            int row, col, num_rows = x_.shape[0]

            vector[vector[vector[Rule]]] filter_trees_all
            vector[vector[Rule]] filtered_trees
            vector[double] col_split_points, expected_values

            const Rule* rule_ptr
            const vector[Rule]* tree_ptr

            double[:,:] values = np.empty((num_rows, self.len_col), dtype=np.float64)


        if back_data is not None:
            back_data_ = self.model.predict(back_data, pred_leaf=True).astype(np.int32)
        else:
            back_data_ = np.empty((0, 0), dtype=np.int32)


        expected_values.resize(self.len_col)
        for col in range(self.len_col):
            expected_values[col] = _expected_value(col, self.trees, back_data_)


        filter_trees_all.resize(self.len_col)
        for col in range(self.len_col):
            filter_trees_all[col] = filter_trees(self.trees, col)
        
        with nogil:
            for col in range(self.len_col):
                filtered_trees = filter_trees_all[col]
                expected_value = expected_values[col]
                num_trees = filtered_trees.size()

                for row in range(num_rows):
                    point = x_[row, col]

                    ensemble_sum = 0.0
                    tree_count = 0.0

                    for i in range(num_trees):
                        tree_ptr = &filtered_trees[i]
                        tree_size = tree_ptr.size()

                        point_col_sum = 0.0
                        point_iter_count = 0.0
                        point_count = 0.0

                        for j in range(tree_size):
                            rule_ptr = &(tree_ptr[0][j])
                            is_valid_rule = check_value(rule_ptr, col, point)

                            if is_valid_rule:
                                rule_val = rule_ptr.value
                                n_count = rule_ptr.count

                                point_col_sum += rule_val * n_count
                                point_count += n_count
                                point_iter_count += 1

                        if point_count > 0:
                            tree_count += 1.0
                            ensemble_sum += (point_col_sum / point_count)

                    values[row, col] = ensemble_sum - expected_value

        return np.asarray(values)

    @boundscheck(False)
    @nonecheck(False)
    @wraparound(False)
    @initializedcheck(False)
    @overflowcheck(False)
    @cdivision(True)
    @infer_types(True)
    cpdef object analyze_feature(self, int col):
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
            str column_name
            object df

        points, mean_values, stds, counts = _analyze_feature(col, self.trees)
                    
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
    cpdef object count_node(self, bint interaction=True):
        if not isinstance(interaction, bool):
            raise ValueError("The 'interaction' parameter must be set explicitly to either True or False.")

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
