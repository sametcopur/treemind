from libcpp.vector cimport vector

from libc.math cimport INFINITY

import pandas as pd

from libcpp.pair cimport pair

from .rule cimport update_leaf_counts
from .utils cimport add_lower_bound, _analyze_feature, filter_class_trees
from .lgb cimport analyze_lightgbm
from .xgb cimport analyze_xgboost
from .cb cimport analyze_catboost

from collections import Counter
from itertools import combinations
import warnings

cdef vector[pair[double, double]] feature_ranges

cdef class Explainer:
    def __init__(self):
        self.trees = vector[vector[Rule]]()
        self.cat_cols = vector[vector[int]]()
        self.cat_indices = vector[int]()
        self.model = None
        self.categorical = None
        self.len_col = -1
        self.columns = None
        self.model_type = "none"
        self.n_classes = 0

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

            # if self.model._Booster__num_class != 1:
            #     raise ValueError("Multiclass lightgbm models are not supported yet.")
            self.n_classes = self.model._Booster__num_class
            self.columns = self.model.feature_name()
            self.len_col = len(self.columns)
            self.model_type = "lightgbm"
            self.categorical = self.model.pandas_categorical
            self.trees, self.cat_cols, self.cat_indices = analyze_lightgbm(self.model, self.len_col)

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
            self.categorical = None
            self.trees, self.cat_cols, self.cat_indices, self.n_classes = analyze_xgboost(self.model, self.len_col)

            if self.cat_indices.size() > 0:
                warnings.warn(
                    "XGBoost models store categorical variables as indices, "
                    "making it impossible to retrieve their original category names.\n"
                    "If a categorical feature or its values are not used in any split, "
                    "they may not appear in the extracted results.\n"
                    "To map indices back to category names, you need to provide the original "
                    "categorical mappings from the training data and include them in the output of analyze_feature."
                )

        elif "catboost" in module_name:
            self.model = model
            if len(self.model.get_cat_feature_indices()) > 0:
                raise ValueError(
                    "Obtaining which category and prior a CTR split belongs to from a catboost model dump is not possible.\n"
                    "For more details, please refer to: https://github.com/catboost/catboost/issues/2414"
                )
            self.categorical = None
            self.len_col = self.model.n_features_in_
            self.columns = model.feature_names_
            self.trees, self.cat_cols, self.cat_indices = analyze_catboost(self.model, self.len_col)
            self.model_type = "catboost"

        # Raise an error if the model is not lightgbm or xgboost
        else:
            raise ValueError("The provided model isn't a lightgbm, xgboost or catboost model. Please provide a supported model type.")

    cdef object prepare_dataframe(self, vector[int] col_indices, 
                    int num_cols, 
                    list column_names,
                    vector[vector[double]]& points,
                    vector[double]& mean_values,
                    vector[double]& ensemble_std,
                    vector[double]& counts,
                    object columns):

        cdef int idx, i, j, ub_index
        cdef bint check
    
        cdef str cats, msg, col_ub
        cdef dict df_dict = {}
        cdef vector[double] row
        cdef object df
        
        # Add points columns
        for i in range(num_cols):
            check = True
            for j in range(self.cat_indices.size()):
                if col_indices[i] == self.cat_indices[j]:
                    if self.categorical is not None:
                        cats = self.categorical[j]
                        df_dict[f"{column_names[i]}"] = [cats[int(row[i])] for row in points]
                    else:
                        df_dict[f"{column_names[i]}"] = [int(row[i]) for row in points]
                    check = False

            if check:
                df_dict[f"{column_names[i]}_ub"] = [row[i] for row in points]
        
        # Add statistics columns
        df_dict.update({
            'value': mean_values,
            'std': ensemble_std,
            'count': counts
        })
        
        df = pd.DataFrame(df_dict)
        
        # Center the values
        if df["count"].sum() == 0:
            df.loc[:, "std"] = df["value"]

            if isinstance(columns, int) or len(columns) == 1:
                msg = f"Feature {columns} was not used in any split by the model."
            else:
                msg = f"One or more of the features {columns} were  not used in any split by the model."

            warnings.warn(msg)
                

        else:
            df.loc[:, "value"] -= (df["value"] * df["count"]).sum() / df["count"].sum()
        
        # # Add lower bounds for all columns
        for col_name in column_names:
            col_ub = f"{col_name}_ub"

            if col_ub in df.columns:
                # 'column_ub' kolonunun indeksini bul
                ub_index = df.columns.get_loc(col_ub)
                
                # 'column_lb' bu indeksin önüne eklenmeli
                add_lower_bound(df, ub_index, col_name)
        
        return df
    
    def analyze_feature(self, columns, *, back_data = None):
        if self.len_col == -1:
            raise ValueError("Explainer(model) must be called before this operation.")
        
        # Convert columns list to vector[int]
        cdef:
            vector[int] col_indices
            int col
            vector[vector[Rule]] trees = self.trees
            vector[vector[Rule]] class_trees
            vector[vector[double]] points
            vector[double] mean_values, ensemble_std, counts
            int num_cols, i
            str col_str

        
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
        
        # Get column names using regular Python list
        column_names = [self.columns[idx] for idx in col_indices]

        # Analyze interactions
        cdef list all_df = []
        for class_idx in range(self.n_classes):
            if self.n_classes == 1:
                class_trees = trees
            else:
                class_trees = filter_class_trees(trees, self.n_classes, class_idx) 
            
            points, mean_values, ensemble_std, counts = _analyze_feature(class_trees, col_indices, self.cat_cols)
            class_df = self.prepare_dataframe(col_indices, num_cols, column_names, points, mean_values, ensemble_std, counts, columns)

            if self.n_classes != 1:
                class_df.loc[:, "class"] = class_idx
            
            all_df.append(class_df)

        df = pd.concat(all_df, ignore_index=True)

        # Closure olmayan biçimiyle yaz
        only_default_split = True
        for col_str in df.columns:
            if col_str not in {"value", "std", "count"} and not col_str.endswith("_lb") and not col_str.endswith("_ub"):
                only_default_split = False
                break

        if df.shape[0] == 1 and only_default_split:
            if isinstance(columns, int) or len(columns) == 1:
                msg = f"Feature {columns} was not used in any split. Values are based on the default range (-inf, +inf)."
            else:
                msg = f"All of the features {columns} were not used in any split. Values are based on the default range (-inf, +inf)."
            
            warnings.warn(msg)

        
        return df


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
                    if rule.lbs[i] != -INFINITY or rule.ubs[i] != INFINITY or rule.cat_flags[i]
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
            