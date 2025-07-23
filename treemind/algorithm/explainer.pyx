from libcpp.vector cimport vector
from libc.math cimport INFINITY
import pandas as pd
from libcpp.pair cimport pair
from .rule cimport update_leaf_counts
from .utils cimport add_lower_bound, _analyze_feature, filter_class_trees
from .lgb cimport analyze_lightgbm
from .xgb cimport analyze_xgboost
from .cb cimport analyze_catboost
from .sk cimport analyze_sklearn
from .perp cimport analyze_perpetual

from collections import Counter
from itertools import combinations
import warnings
from tqdm import tqdm
import numpy as np

cdef vector[pair[float, float]] feature_ranges

cdef class Result:
    """Result class to hold feature interaction analysis statistics"""
    
    def __init__(self):
        self.data = {}
        self.degree = 0
        self.n_classes = 0
        self.feature_names = []
        self.model_type = ""
    
    def __repr__(self) -> str:
        return f"Result(degree={self.degree}, interactions={len(self.data)}, classes={self.n_classes})"
        
    def __getitem__(self, key):
        """
        Get statistics for feature interaction using Python indexing
        
        Parameters:
        -----------
        key : int, list, tuple
            - Single int: get individual feature stats (for degree=1)
            - List/tuple of ints: get interaction stats for those features
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with feature interaction statistics. 
            For multi-class models, includes 'class' column indicating the class.
        """
        if self.data == {}:
            raise ValueError("No data available. Please run the explain method first.")

        # Handle single integer (individual feature)
        if isinstance(key, int):
            if self.degree == 1:
                requested_key = (key,)
            else:
                raise ValueError(f"Single index only valid for degree=1, current degree={self.degree}")

            if key < 0 or key >= len(self.feature_names):
                raise IndexError(f"Index {key} out of bounds. Must be between 0 and {len(self.feature_names) - 1}.")

        # Handle list or tuple (feature interactions)
        elif isinstance(key, (list, tuple)):
            if len(key) != self.degree:
                raise ValueError(f"Index length ({len(key)}) must match result degree ({self.degree})")

            requested_key = tuple(key)

            # Additional validation: prevent same indices when degree >= 2
            if self.degree >= 2 and len(set(requested_key)) == 1:
                raise ValueError(f"All indices are the same ({requested_key}) and degree >= 2 is not allowed.")

            for idx in requested_key:
                if idx < 0 or idx >= len(self.feature_names):
                    raise IndexError(f"Index {idx} out of bounds. Must be between 0 and {len(self.feature_names) - 1}.")

        else:
            raise TypeError("Index must be int, list, or tuple")
        
        # Find the data by checking all possible permutations of the key
        stored_data = None
        stored_key = None
        
        # Check if exact key exists
        if requested_key in self.data:
            stored_data = self.data[requested_key]
            stored_key = requested_key
        else:
            # Check all permutations to find the stored version
            from itertools import permutations
            for perm in permutations(requested_key):
                if perm in self.data:
                    stored_data = self.data[perm]
                    stored_key = perm
                    break
        
        if stored_data is None:
            return None
        
        # If requested order is different from stored order, reorder the DataFrames
        if requested_key != stored_key:
            stored_data = self._reorder_dataframes(stored_data, stored_key, requested_key)
        
        # Convert class-wise data to single DataFrame
        return self._combine_class_dataframes(stored_data)

    cdef _combine_class_dataframes(self, dict class_data):
        """
        Combine class-wise DataFrames into a single DataFrame with optional 'class' column
        """
        cdef list dataframes = []
        cdef int class_idx
        cdef object df, combined_df
        
        # If single class, return the DataFrame directly
        if self.n_classes == 1:
            return class_data[0]
        
        # For multi-class, combine all class DataFrames
        for class_idx in sorted(class_data.keys()):
            df = class_data[class_idx].copy()
            df['class'] = class_idx
            dataframes.append(df)
        
        # Concatenate all DataFrames
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Reorder columns to put 'class' first
        cols =  [col for col in combined_df.columns if col != 'class'] + ['class']
        combined_df = combined_df[cols]
        
        return combined_df

    cdef _reorder_dataframes(self, dict class_data, tuple stored_key, tuple requested_key):
        """
        Reorder DataFrame columns according to requested key order
        """
        cdef dict reordered_data = {}
        cdef int class_idx
        cdef object df, reordered_df
        
        # Create mapping from stored order to requested order
        cdef list reorder_mapping = []
        for req_idx in requested_key:
            stored_position = stored_key.index(req_idx)
            reorder_mapping.append(stored_position)
        
        for class_idx, df in class_data.items():
            reordered_df = self._reorder_single_dataframe(df, stored_key, requested_key, reorder_mapping)
            reordered_data[class_idx] = reordered_df
        
        return reordered_data
    
    cdef _reorder_single_dataframe(self, object df, tuple stored_key, tuple requested_key, list reorder_mapping):
        """
        Reorder a single DataFrame's columns according to the requested feature order
        """
        cdef list new_columns = []
        cdef list feature_columns = []
        cdef list other_columns = []
        cdef str col
        cdef int i, req_idx, stored_idx
        cdef str feature_name
        cdef list current_feature_cols
        
        # Separate feature columns from other columns (value, std, count)
        for col in df.columns:
            if col in ['value', 'std', 'count']:
                other_columns.append(col)
            else:
                feature_columns.append(col)
        
        # Reorder feature columns according to requested order
        for req_idx in requested_key:
            feature_name = self.feature_names[req_idx]
            current_feature_cols = []
            
            # Find all columns belonging to this feature
            for col in feature_columns:
                # Check if column belongs to this feature
                if (col == feature_name or                           # Categorical: exact match
                    col == f"{feature_name}_lb" or                   # Numerical: lower bound
                    col == f"{feature_name}_ub"):                    # Numerical: upper bound
                    current_feature_cols.append(col)
            
            # Sort feature columns in consistent order: _lb, _ub, then base name
            # This handles both cases:
            # - Categorical: just feature_name
            # - Numerical: feature_name_lb, feature_name_ub
            current_feature_cols.sort(key=lambda x: (
                not x.endswith('_lb'),    # _lb comes first (False = 0)
                not x.endswith('_ub'),    # _ub comes second (False = 0) 
                x                         # Then alphabetical for any remaining
            ))
            
            new_columns.extend(current_feature_cols)
        
        # Add other columns at the end
        new_columns.extend(other_columns)
        
        # Reorder the DataFrame
        return df[new_columns]


    # Result sınıfı içine ekleyin
    def importance(self, bint combine_classes=False):
        cdef list rows = []
        cdef tuple feat_key
        cdef dict class_data
        cdef int cls
        cdef object df
        cdef float total_cnt, I_abs, num

        if self.data == {}:
            raise ValueError("No data available. Please run the explain method first.")

        if self.n_classes == 1 and combine_classes:
            raise ValueError("Combined class importance is not supported for single-class models. Please use `combine_classes=False`.")

        for feat_key, class_data in self.data.items():

            def _calc_I_abs(df):
                total = df['count'].sum()
                if total == 0:
                    return float('nan')
                mu = (df['value'] * df['count']).sum() / total
                I_abs = ((df['value'] - mu).abs() * df['count']).sum() / total
                return I_abs

            if self.n_classes == 1 or combine_classes:
                # Tek sınıf veya birleştirilmiş çoklu sınıf
                total_cnt = 0.0
                num = 0.0
                for cls, df in class_data.items():
                    cnt = df['count'].sum()
                    if cnt == 0:
                        continue
                    Ia   = _calc_I_abs(df)
                    num += Ia * cnt
                    total_cnt += cnt
                I_abs = num / total_cnt if total_cnt else float('nan')

                row = {f'feature_{i}': self.feature_names[idx]
                    for i, idx in enumerate(feat_key)}
                row['importance'] = I_abs
                rows.append(row)

            else:
                # Sınıf bazlı detaylı çıktı
                for cls, df in class_data.items():
                    I_abs = _calc_I_abs(df)
                    row = {f'feature_{i}': self.feature_names[idx]
                        for i, idx in enumerate(feat_key)}
                    row['importance'] = I_abs
                    row['class'] = cls
                    rows.append(row)

        return pd.DataFrame(rows).sort_values(by='importance', ascending=False).reset_index(drop=True)

    
    def __contains__(self, key):
        """Check if a feature interaction exists in the results"""
        try:
            return self[key] is not None
        except (ValueError, TypeError):
            return False
    
    def __len__(self):
        """Return number of feature interactions"""
        return len(self.data)
    
    def __iter__(self):
        """Iterate over feature combinations"""
        return iter(self.data.keys())
    
    def keys(self):
        """Return all feature combinations"""
        return self.data.keys()
    
    def values(self):
        """Return all statistics"""
        return self.data.values()
    
    def items(self):
        """Return feature combinations and their statistics"""
        return self.data.items()
    
cdef class Explainer:
    def __init__(self, object model):
        self.trees = vector[vector[Rule]]()
        self.cat_cols = vector[vector[int]]()
        self.cat_indices = vector[int]()
        self.model = None
        self.categorical = None
        self.len_col = -1
        self.columns = None
        self.model_type = "none"
        self.n_classes = 0

        self._extract_model(model)

    def __repr__(self) -> str:
        return f"Explainer(model={self.model_type})"

    def _extract_model(self, model):

        module_name = model.__module__

        # Handle lightgbm models
        if "lightgbm" in module_name: 
            if "basic" not in module_name:
                self.model = model.booster_
            else:
                self.model = model

            self.n_classes = self.model._Booster__num_class
            self.columns = self.model.feature_name()
            self.len_col = len(self.columns)
            self.model_type = "lightgbm"
            self.categorical = self.model.pandas_categorical
            self.trees, self.cat_cols, self.cat_indices = analyze_lightgbm(self.model, self.len_col)

        elif "sklearn" in module_name:
            self.model = model
            self.categorical = None

            model_class = model.__class__.__name__
            is_histgb = "HistGradientBoosting" in model_class

            # Feature length
            if hasattr(model, 'n_features_in_'):
                self.len_col = model.n_features_in_
            elif hasattr(model, 'n_features_'):
                self.len_col = model.n_features_
            elif is_histgb and hasattr(model, '_predictors'):
                # For HistGradientBoosting, infer from the nodes structure
                if isinstance(model._predictors[0], list):
                    self.len_col = max(p.nodes['feature_idx'].max() for p in model._predictors[0]) + 1
                else:
                    self.len_col = model._predictors[0].nodes['feature_idx'].max() + 1
            else:
                # Fallback: try to get from tree structure
                if hasattr(model, 'tree_'):
                    self.len_col = model.tree_.n_features
                elif hasattr(model, 'estimators_'):
                    model_class_name = model.__class__.__name__
                    if 'GradientBoosting' in model_class_name:
                        first_estimator = model.estimators_[0, 0]
                        self.len_col = (
                            first_estimator.tree_.n_features
                            if hasattr(first_estimator, 'tree_')
                            else first_estimator.n_features_
                        )
                    else:
                        if hasattr(model.estimators_[0], 'tree_'):
                            self.len_col = model.estimators_[0].tree_.n_features
                        else:
                            self.len_col = model.estimators_[0].n_features_
                else:
                    raise ValueError("Could not determine number of features from sklearn model")

            # Feature names
            if hasattr(model, 'feature_names_in_'):
                self.columns = list(model.feature_names_in_)
            else:
                self.columns = [f"Feature_{i}" for i in range(self.len_col)]

            # Class count
            if hasattr(model, 'n_classes_'):
                self.n_classes = 1 if model.n_classes_ == 2 else model.n_classes_
            elif is_histgb and hasattr(model, 'classes_'):
                self.n_classes = 1 if len(model.classes_) <= 2 else len(model.classes_)
            else:
                self.n_classes = 1  # Regression or binary classification

            self.model_type = "sklearn"
            self.trees, self.cat_cols, self.cat_indices = analyze_sklearn(self.model, self.len_col, self.n_classes)

        elif "perpetual" in module_name:
            self.model = model
            self.len_col = model.n_features_
            self.columns = model.feature_names_in_
            self.model_type = "perpetual"
            self.categorical = list(model.cat_mapping.values())
            self.n_classes = 1 if len(model.classes_) <= 2 else len(model.classes_)

            self.trees, self.cat_cols, self.cat_indices = analyze_perpetual(self.model)

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
            self.trees, self.cat_cols, self.cat_indices, self.n_classes = analyze_catboost(self.model, self.len_col)
            self.model_type = "catboost"

        # Raise an error if the model is not lightgbm or xgboost
        else:
            raise ValueError("The provided model isn't a lightgbm, xgboost or catboost model. Please provide a supported model type.")

    cdef object prepare_dataframe(self, vector[int] col_indices, 
                    int num_cols, 
                    list column_names,
                    vector[vector[float]]& points,
                    vector[float]& mean_values,
                    vector[float]& ensemble_std,
                    vector[float]& counts,
                    object columns):

        cdef int idx, i, j, ub_index
        cdef bint check
    
        cdef str msg, col_ub
        cdef dict df_dict = {}
        cdef vector[float] row
        cdef object df
        cdef list cats
        
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
            'value': np.asarray(mean_values, dtype=np.float32),
            'std': np.asarray(ensemble_std, dtype=np.float32),
            'count': np.asarray(counts, dtype=np.float32)
        })
        df = pd.DataFrame(df_dict, dtype=np.float32)

        # Center the values
        if df["count"].sum() == 0:
            df.loc[:, "std"] = df["value"]

            if isinstance(columns, int) or len(columns) == 1:
                msg = f"Feature {columns} was not used in any split by the model."
            else:
                msg = f"One or more of the features {columns} were not used in any split by the model."

            warnings.warn(msg)
        else:
            adjustment = (df["value"] * df["count"]).sum() / df["count"].sum()
            df.loc[:, "value"] = (df["value"] - adjustment).astype(np.float32)
        
        # Add lower bounds for all columns
        for col_name in column_names:
            col_ub = f"{col_name}_ub"

            if col_ub in df.columns:
                ub_index = df.columns.get_loc(col_ub)
                add_lower_bound(df, ub_index, col_name)
        
        return df
    
    def explain(self, int degree, *, back_data=None):

        if self.len_col == -1:
            raise ValueError("Explainer(model) must be called before this operation.")
        
        if degree >= self.len_col:
            raise ValueError("'degree' cannot be greater than or equal to the total number of columns used to train the model.")

        if degree < 1:
            raise ValueError("The 'degree' parameter must be a positive integer.")

        cdef:
            vector[vector[Rule]] trees = self.trees
            vector[vector[Rule]] class_trees
            Result result = Result()
            list feature_combinations
            tuple feature_combo
            vector[int] col_indices
            int col
            vector[vector[float]] points
            vector[float] mean_values, ensemble_std, counts
            int num_cols, class_idx
            list column_names
            object class_df

        # Generate all feature combinations of specified degree
        feature_combinations = list(combinations(range(self.len_col), degree))
        
        if not feature_combinations:
            warnings.warn(f"No feature combinations of degree {degree} possible with {self.len_col} features.")
            return result

        # Update trees with background data if provided
        if back_data is not None:
            trees = update_leaf_counts(trees, self.model, back_data, self.model_type)

        # Set result metadata
        result.degree = degree
        result.n_classes = self.n_classes
        result.feature_names = self.columns
        result.model_type = self.model_type

        # Analyze each feature combination
        for feature_combo in tqdm(feature_combinations, desc="Analyzing features"):
            # Convert feature combination to vector[int]
            col_indices.clear()
            for col in feature_combo:
                col_indices.push_back(col)
            
            num_cols = len(feature_combo)
            column_names = [self.columns[idx] for idx in feature_combo]
            
            # Store class-wise statistics for this feature combination
            class_stats = {}
            
            # Analyze for each class
            for class_idx in range(self.n_classes):
                if self.n_classes == 1:
                    class_trees = trees
                else:
                    class_trees = filter_class_trees(trees, self.n_classes, class_idx)
                
                points, mean_values, ensemble_std, counts = _analyze_feature(class_trees, col_indices, self.cat_cols)
                class_df = self.prepare_dataframe(col_indices, num_cols, column_names, points, mean_values, ensemble_std, counts, feature_combo)
                
                class_stats[class_idx] = class_df
            
            # Store in result
            result.data[feature_combo] = class_stats

        return result

    cpdef object count_node(self, int degree=2):
        if self.len_col == -1:
            raise ValueError("Explainer(model) must be called before this operation.")

        if degree >= self.len_col:
            raise ValueError("'degree' cannot be greater than or equal to the total number of columns used to train the model.")

        if degree < 1:
            raise ValueError("The 'degree' parameter must be a positive integer.")

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

                if n >= degree:
                    for comb in combinations(finite_indices, degree):
                        combination_counts[comb] += 1

        # Create column names dynamically based on order
        columns = [f"column{i+1}_index" for i in range(degree)]
        columns.append("count")

        # Prepare data for DataFrame
        data = [(*(comb), count) for comb, count in combination_counts.items()]

        df = pd.DataFrame(data, columns=columns)

        return df.sort_values("count", ascending=False).reset_index(drop=True)