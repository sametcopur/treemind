# Standard library cimports
from cython.parallel cimport prange
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.algorithm cimport sort, unique
from libcpp.pair cimport pair
from libc.math cimport INFINITY, sqrt, log, exp

cimport numpy as cnp
import numpy as np
import polars as pl
import json
import tempfile
import os
import importlib
from collections import Counter
from itertools import combinations, permutations
import warnings
from tqdm import tqdm

# Shared typedefs
ctypedef pair[float, float] RangePair
ctypedef pair[double, double] RangePairD

# Include all internal modules (ORDER MATTERS)
include "_rule.pxi"
include "_utils.pxi"
include "_lgb.pxi"
include "_xgb.pxi"
include "_rule_leaf.pxi"
include "_cb.pxi"
include "_sk.pxi"
include "_perp.pxi"

cdef class Result:
    """Result class to hold feature interaction analysis statistics"""
    
    def __init__(self):
        self._data = {}
        self.degree = 0
        self.n_classes = 0
        self.feature_names = []
        self.model_type = ""
    
    def __repr__(self) -> str:
        return f"Result(degree={self.degree}, interactions={len(self._data)}, classes={self.n_classes})"
        
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
        if self._data == {}:
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
        if requested_key in self._data:
            stored_data = self._data[requested_key]
            stored_key = requested_key
        else:
            # Check all permutations to find the stored version
            for perm in permutations(requested_key):
                if perm in self._data:
                    stored_data = self._data[perm]
                    stored_key = perm
                    break
        
        if stored_data is None:
            return None
        
        # If requested order is different from stored order, reorder points columns
        if requested_key != stored_key:
            reorder_mapping = [stored_key.index(idx) for idx in requested_key]
            stored_data = {cls: (pts[:, reorder_mapping], m, s, c)
                          for cls, (pts, m, s, c) in stored_data.items()}

        # Build DataFrame lazily from sparse storage
        return self._build_dataframe(stored_data, requested_key)

    cdef _build_dataframe(self, dict class_data, tuple feature_key):
        """Build polars DataFrame from sparse numpy arrays on demand."""
        cdef list dataframes = []
        cdef int class_idx, i, j
        cdef str col_name, col_ub
        cdef dict df_dict

        # Build O(1) lookup for categorical indices
        cdef unordered_set[int] cat_set
        cdef dict cat_position = {}
        for j in range(self.cat_indices.size()):
            cat_set.insert(self.cat_indices[j])
            cat_position[self.cat_indices[j]] = j

        cdef list column_names = [self.feature_names[idx] for idx in feature_key]
        cdef int num_cols = len(feature_key)

        for class_idx in sorted(class_data.keys()):
            points, means, stds, counts = class_data[class_idx]

            df_dict = {}

            # Add feature columns
            for i in range(num_cols):
                col_idx = feature_key[i]
                if cat_set.count(col_idx) > 0:
                    j = cat_position[col_idx]
                    if self.categorical is not None:
                        cats = self.categorical[j]
                        df_dict[column_names[i]] = [cats[int(v)] for v in points[:, i]]
                    else:
                        df_dict[column_names[i]] = points[:, i].astype(np.int32)
                else:
                    df_dict[f"{column_names[i]}_ub"] = points[:, i]

            # Add statistics
            df_dict['value'] = means
            df_dict['std'] = stds
            df_dict['count'] = counts

            df = pl.DataFrame(df_dict)

            # Center values
            count_sum = float(df["count"].sum())
            if count_sum > 0:
                adjustment = float((df["value"] * df["count"]).sum()) / count_sum
                df = df.with_columns((pl.col("value") - adjustment).cast(pl.Float32).alias("value"))

            # Add lower bounds for continuous columns
            for col_name in column_names:
                col_ub = f"{col_name}_ub"
                if col_ub in df.columns:
                    ub_values = df[col_ub].to_numpy()
                    unique_values = np.sort(np.unique(ub_values))
                    lower_bounds = np.empty_like(unique_values, dtype=np.float32)
                    lower_bounds[0] = -np.inf
                    lower_bounds[1:] = unique_values[:len(unique_values) - 1]
                    lb_map = dict(zip(unique_values, lower_bounds))
                    lb_col = np.array([lb_map[v] for v in ub_values], dtype=np.float32)
                    ub_idx = df.columns.index(col_ub)
                    cols_before = df.columns[:ub_idx]
                    df = df.select(
                        [pl.col(c) for c in cols_before] +
                        [pl.Series(f"{col_name}_lb", lb_col)] +
                        [pl.col(c) for c in df.columns[ub_idx:]]
                    )

            if self.n_classes > 1:
                df = df.with_columns(pl.lit(class_idx).alias("class"))

            dataframes.append(df)

        if len(dataframes) == 1:
            return dataframes[0]

        combined = pl.concat(dataframes)
        cols = [c for c in combined.columns if c != 'class'] + ['class']
        return combined.select(cols)


    def importance(self, bint combine_classes=False):
        """
        Calculate feature importance with interaction correction.
        Subtracts single-feature effects from interactions to isolate true interaction strength.
        """
        if self._data == {}:
            raise ValueError("No data available. Please run the explain method first.")

        if self.n_classes == 1 and combine_classes:
            raise ValueError("Combined class importance is not supported for single-class models.")

        single_importances = self._compute_single_importances(combine_classes)
        rows = self._compute_importance_rows(single_importances, combine_classes)
        return pl.DataFrame(rows).sort("importance", descending=True)

    @staticmethod
    def _calc_abs_importance(sparse_tuple):
        """Compute mean absolute deviation weighted by count from sparse (pts, means, stds, counts) tuple."""
        _, values, _, counts = sparse_tuple
        total = counts.sum()
        if total == 0:
            return float('nan')
        mu = (values * counts).sum() / total
        return float((np.abs(values - mu) * counts).sum() / total)

    def _compute_single_importances(self, bint combine_classes):
        """Compute importance scores for all single (degree-1) features."""
        cdef dict single_importances = {}
        cdef tuple feat_key
        cdef dict class_data
        cdef int cls, feat_idx
        cdef float total_cnt, weighted_sum, abs_importance

        for feat_key, class_data in self._data.items():
            if len(feat_key) != 1:
                continue
            feat_idx = feat_key[0]

            if self.n_classes == 1 or combine_classes:
                total_cnt = 0.0
                weighted_sum = 0.0
                for cls, sparse in class_data.items():
                    cnt = sparse[3].sum()  # counts array
                    if cnt == 0:
                        continue
                    abs_importance = self._calc_abs_importance(sparse)
                    weighted_sum += abs_importance * cnt
                    total_cnt += cnt
                single_importances[feat_idx] = weighted_sum / total_cnt if total_cnt else 0.0
            else:
                single_importances[feat_idx] = {}
                for cls, sparse in class_data.items():
                    single_importances[feat_idx][cls] = self._calc_abs_importance(sparse)

        return single_importances

    def _compute_importance_rows(self, dict single_importances, bint combine_classes):
        """Build importance rows with interaction correction applied."""
        cdef list rows = []
        cdef tuple feat_key
        cdef dict class_data
        cdef int cls
        cdef float total_cnt, weighted_sum, raw_importance, abs_importance, sum_single

        for feat_key, class_data in self._data.items():
            if self.n_classes == 1 or combine_classes:
                total_cnt = 0.0
                weighted_sum = 0.0
                for cls, sparse in class_data.items():
                    cnt = sparse[3].sum()
                    if cnt == 0:
                        continue
                    abs_importance = self._calc_abs_importance(sparse)
                    weighted_sum += abs_importance * cnt
                    total_cnt += cnt
                raw_importance = weighted_sum / total_cnt if total_cnt else 0.0

                if len(feat_key) > 1:
                    sum_single = sum([single_importances.get(idx, 0.0) for idx in feat_key])
                    abs_importance = max(0.0, raw_importance - sum_single)
                else:
                    abs_importance = raw_importance

                row = {f'feature_{i}': self.feature_names[idx] for i, idx in enumerate(feat_key)}
                row['importance'] = abs_importance
                rows.append(row)

            else:
                for cls, sparse in class_data.items():
                    raw_importance = self._calc_abs_importance(sparse)

                    if len(feat_key) > 1:
                        sum_single = sum([single_importances.get(idx, {}).get(cls, 0.0) for idx in feat_key])
                        abs_importance = max(0.0, raw_importance - sum_single)
                    else:
                        abs_importance = raw_importance

                    row = {f'feature_{i}': self.feature_names[idx] for i, idx in enumerate(feat_key)}
                    row['importance'] = abs_importance
                    row['class'] = cls
                    rows.append(row)

        return rows


    def __contains__(self, key):
        """Check if a feature interaction exists in the results"""
        try:
            return self[key] is not None
        except (ValueError, TypeError):
            return False
    
    def __len__(self):
        """Return number of feature interactions"""
        return len(self._data)
    
    def __iter__(self):
        """Iterate over feature combinations"""
        return iter(self._data.keys())

    @property
    def data(self):
        """Return the internal data dictionary"""
        return self._data
    
    
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
        """Dispatch model to the appropriate initialization method."""
        module_name = model.__module__

        if "lightgbm" in module_name:
            self._init_lightgbm(model)
        elif "sklearn" in module_name:
            self._init_sklearn(model)
        elif "perpetual" in module_name:
            self._init_perpetual(model)
        elif "xgboost" in module_name:
            self._init_xgboost(model)
        elif "catboost" in module_name:
            self._init_catboost(model)
        else:
            raise ValueError(
                "The provided model isn't a supported model type. "
                "Please provide a lightgbm, xgboost, catboost, sklearn, or perpetual model."
            )

    def _init_lightgbm(self, model):
        if "basic" not in model.__module__:
            self.model = model.booster_
        else:
            self.model = model

        self.n_classes = self.model._Booster__num_class
        self.columns = self.model.feature_name()
        self.len_col = len(self.columns)
        self.model_type = "lightgbm"
        self.categorical = self.model.pandas_categorical
        self.trees, self.cat_cols, self.cat_indices = analyze_lightgbm(self.model, self.len_col)

    def _init_sklearn(self, model):
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
            if isinstance(model._predictors[0], list):
                self.len_col = max(p.nodes['feature_idx'].max() for p in model._predictors[0]) + 1
            else:
                self.len_col = model._predictors[0].nodes['feature_idx'].max() + 1
        else:
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
            self.n_classes = 1

        self.model_type = "sklearn"
        self.trees, self.cat_cols, self.cat_indices = analyze_sklearn(self.model, self.len_col, self.n_classes)

    def _init_perpetual(self, model):
        self.model = model
        self.len_col = model.n_features_
        self.columns = model.feature_names_in_
        self.model_type = "perpetual"
        self.categorical = list(model.cat_mapping.values())
        self.n_classes = 1 if len(model.classes_) <= 2 else len(model.classes_)
        self.trees, self.cat_cols, self.cat_indices = analyze_perpetual(self.model)

    def _init_xgboost(self, model):
        if "core" not in model.__module__:
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

    def _init_catboost(self, model):
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

    def explain(self, int degree, *, back_data=None):

        if self.len_col == -1:
            raise ValueError("Explainer(model) must be called before this operation.")

        if degree >= self.len_col:
            raise ValueError("'degree' cannot be greater than or equal to the total number of columns used to train the model.")

        if degree < 1:
            raise ValueError("The 'degree' parameter must be a positive integer.")

        cdef:
            vector[vector[Rule]] trees = self.trees
            Result result = Result()
            list feature_combinations
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
            raise ValueError(
                "Background data is disabled in the current implementation. For now, please use the model's "
                "training data to explain feature interactions. Next versions will support background data."
            )

        # Set result metadata
        result.degree = degree
        result.n_classes = self.n_classes
        result.feature_names = self.columns
        result.model_type = self.model_type
        result.cat_indices = self.cat_indices
        result.categorical = self.categorical

        cdef int n_combos = <int>len(feature_combinations)
        cdef int n_classes = self.n_classes
        cdef int n_tasks = n_combos * n_classes
        cdef int task_idx, combo_idx, cls_idx, col, i, feat_idx

        # Pre-compute col_indices for all combinations
        cdef vector[vector[int]] all_col_indices = vector[vector[int]](n_combos)
        for i in range(n_combos):
            for col in feature_combinations[i]:
                all_col_indices[i].push_back(col)

        # Pre-compute class trees (sequential)
        cdef vector[vector[vector[Rule]]] all_class_trees
        if n_classes > 1:
            all_class_trees.resize(n_classes)
            for class_idx in range(n_classes):
                all_class_trees[class_idx] = filter_class_trees(trees, n_classes, class_idx)

        cdef vector[vector[float]] split_cache
        cdef vector[vector[vector[float]]] class_split_caches
        if n_classes == 1:
            split_cache = vector[vector[float]](self.len_col)
            for feat_idx in range(self.len_col):
                split_cache[feat_idx] = get_split_point(trees, feat_idx)
        else:
            class_split_caches.resize(n_classes)
            for class_idx in range(n_classes):
                class_split_caches[class_idx] = vector[vector[float]](self.len_col)
                for feat_idx in range(self.len_col):
                    class_split_caches[class_idx][feat_idx] = get_split_point(all_class_trees[class_idx], feat_idx)

        # Cache self attributes for nogil access
        cdef vector[vector[int]] cat_cols_local = self.cat_cols

        # Process in batches to control memory
        cdef int BATCH_SIZE = 128
        cdef int batch_start, batch_end, batch_tasks, local_idx

        cdef vector[vector[vector[float]]] batch_points
        cdef vector[vector[float]] batch_means
        cdef vector[vector[float]] batch_stds
        cdef vector[vector[float]] batch_counts

        for batch_start in tqdm(range(0, n_tasks, BATCH_SIZE), desc="Analyzing features"):
            batch_end = min(batch_start + BATCH_SIZE, n_tasks)
            batch_tasks = batch_end - batch_start

            # Allocate output for this batch only
            batch_points = vector[vector[vector[float]]](batch_tasks)
            batch_means = vector[vector[float]](batch_tasks)
            batch_stds = vector[vector[float]](batch_tasks)
            batch_counts = vector[vector[float]](batch_tasks)

            # Parallel compute for this batch
            with nogil:
                for local_idx in prange(batch_tasks, schedule='dynamic'):
                    task_idx = batch_start + local_idx
                    combo_idx = task_idx // n_classes
                    cls_idx = task_idx % n_classes

                    if n_classes == 1:
                        _analyze_feature_nogil(
                            trees, all_col_indices[combo_idx], cat_cols_local, split_cache,
                            batch_points[local_idx], batch_means[local_idx],
                            batch_stds[local_idx], batch_counts[local_idx])
                    else:
                        _analyze_feature_nogil(
                            all_class_trees[cls_idx], all_col_indices[combo_idx], cat_cols_local, class_split_caches[cls_idx],
                            batch_points[local_idx], batch_means[local_idx],
                            batch_stds[local_idx], batch_counts[local_idx])

            # Store results as sparse numpy arrays (filter zero-count rows)
            for local_idx in range(batch_tasks):
                task_idx = batch_start + local_idx
                combo_idx = task_idx // n_classes
                cls_idx = task_idx % n_classes
                feature_combo = feature_combinations[combo_idx]

                counts_np = np.asarray(batch_counts[local_idx], dtype=np.float32)
                mask = counts_np > 0

                pts_raw = np.asarray(batch_points[local_idx], dtype=np.float32)
                pts_sparse = pts_raw[mask]
                means_sparse = np.asarray(batch_means[local_idx], dtype=np.float32)[mask]
                stds_sparse = np.asarray(batch_stds[local_idx], dtype=np.float32)[mask]
                counts_sparse = counts_np[mask]

                if feature_combo not in result._data:
                    result._data[feature_combo] = {}
                result._data[feature_combo][cls_idx] = (pts_sparse, means_sparse, stds_sparse, counts_sparse)

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
        rows = [{columns[i]: comb[i] for i in range(len(comb))} | {"count": count}
                for comb, count in combination_counts.items()]

        return pl.DataFrame(rows).sort("count", descending=True)
