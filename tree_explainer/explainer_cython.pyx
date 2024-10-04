from libcpp.vector cimport vector
from libcpp.algorithm cimport sort, max_element, min_element, unique
from libc.math cimport INFINITY

from cython cimport boundscheck, wraparound

import numpy as np
cimport numpy as cnp
import pandas as pd

@boundscheck(False)
@wraparound(False)
cdef inline double max(double a, double b):
    return a if a > b else b

@boundscheck(False)
@wraparound(False)
cdef inline double min(double a, double b):
    return a if a < b else b


@boundscheck(False)
@wraparound(False)
cdef void _traverse_tree(object tree, str node_index, dict feature_ranges, vector[Rule]& rules, int tree_index, int len_col):
    cdef object node
    cdef int int_node_index, feature_index
    cdef Rule rule
    cdef double lb, ub, threshold
    cdef dict left_ranges, right_ranges

    # Fetch the current node
    node = tree[tree["node_index"] == node_index].iloc[0]

    # If it is a leaf node
    if pd.isna(node["split_feature"]):
        int_node_index = int(node_index[node_index.find("L") + 1:])
        rule = create_rule(len_col, tree_index, int_node_index)

        # Update rule with feature ranges
        for feature_index, (lb, ub) in feature_ranges.items():
            update_rule(&rule, feature_index, lb, ub)


        # Assign the node value to the rule
        rule.value = node["value"]
        rules.push_back(rule)
        return

    # Extract feature index and threshold for splitting
    feature_index = int(node["split_feature"].replace("Column_", ""))
    threshold = node["threshold"]

    # Copy ranges for left and right splits
    left_ranges = feature_ranges.copy()
    left_ranges[feature_index] = (
        left_ranges.get(feature_index, (-INFINITY, INFINITY))[0],
        threshold,
    )

    right_ranges = feature_ranges.copy()
    right_ranges[feature_index] = (
        threshold,
        right_ranges.get(feature_index, (-INFINITY, INFINITY))[1],
    )

    # Recursive calls for left and right child nodes
    _traverse_tree(tree, node["left_child"], left_ranges, rules, tree_index, len_col)
    _traverse_tree(tree, node["right_child"], right_ranges, rules, tree_index, len_col)


cdef object _replace_inf(object data, str column_name):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] unique_points_ = np.asarray(data[column_name].unique(), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] unique_points = np.sort(unique_points_[np.logical_not(np.isinf(unique_points_))])
    cdef double max_main = unique_points.max()
    cdef double difference_main = max_main - (unique_points[-2] if unique_points.size > 1 else max_main)

    data.loc[np.isinf(data[column_name]), column_name] = max_main + difference_main

    return data

@boundscheck(False)
@wraparound(False)
cdef int compare_rules(const Rule& a, const Rule& b):
    return a.leaf_index < b.leaf_index

@boundscheck(False)
@wraparound(False)
cdef double _find_mean(vector[vector[double]] tree_results) noexcept nogil:
    """
    Calculates the average of the sums of combinations in a memory-efficient way for large inputs.

    Parameters
    ----------
    tree_results : vector[vector[float]]
        A vector of vectors where each inner vector contains float values representing possible
        outcomes from a tree. The function calculates the average over all possible combinations
        of one element from each vector.

    Returns
    -------
    average : float
        The average of the sums of all possible combinations of one element from each sublist.
    """
    cdef double sums = 0.0
    cdef vector[double] sublist
    cdef double sublist_mean
    cdef double total
    cdef int i
    cdef int sublist_size 

    # Calculate the product of the averages of each sublist
    for sublist in tree_results:
        sublist_size = sublist.size()
        total = 0.0
        for i in range(sublist_size):
            total += sublist[i]
        
        sums += total / sublist_size
    
    return sums

@boundscheck(False)
@wraparound(False)
cdef tuple find_min_max(vector[vector[double]] tree_results):
    """
    Finds the minimum and maximum possible values from tree combinations using C++ vectors.

    Parameters
    ----------
    tree_results : vector[vector[double]]
        A C++ vector of vectors where each inner vector represents possible outcomes from a tree.
        Each inner vector contains float values corresponding to the outputs of that tree.

    Returns
    -------
    min_value : double
        The minimum possible value obtained by summing the minimum value from each vector.

    max_value : double
        The maximum possible value obtained by summing the maximum value from each vector.
    """
    cdef double min_value = 0.0
    cdef double max_value = 0.0
    cdef int i
    cdef vector[double] vector_i

    # Iterate through each vector in the tree_results
    for i in range(tree_results.size()):
        vector_i = tree_results[i]

        if vector_i.size() > 1:
            # Use dereference() to properly get the values from the iterators
            min_value += <double>min_element(vector_i.begin(), vector_i.end())[0]
            max_value += <double>max_element(vector_i.begin(), vector_i.end())[0]
        else:
            min_value += vector_i[0]
            max_value += vector_i[0]

    return min_value, max_value


cdef struct Rule:
    int len_col
    int tree_index
    int leaf_index
    vector[double] lbs  # Lower bounds for each feature
    vector[double] ubs  # Upper bounds for each feature
    double value   # Predicted value associated with this rule


@boundscheck(False)
@wraparound(False)
cdef Rule create_rule(int len_col, int tree_index, int leaf_index):
    """
    Initializes a new Rule struct with the given parameters.

    Parameters
    ----------
    len_col : int
        The number of features.
    tree_index : int
        Index of the tree in the ensemble.
    leaf_index : int
        Index of the leaf in the tree.

    Returns
    -------
    Rule
        The initialized Rule struct.
    """
    cdef Rule rule
    rule.len_col = len_col
    rule.tree_index = tree_index
    rule.leaf_index = leaf_index
    rule.lbs = vector[double](len_col, -INFINITY)
    rule.ubs = vector[double](len_col, INFINITY)
    rule.value = np.nan
    return rule

@boundscheck(False)
@wraparound(False)
cdef void update_rule(Rule* rule, int index, double lb, double ub):
    """
    Updates the lower and upper bounds for a feature at the specified index.

    Parameters
    ----------
    rule : Rule*
        The Rule struct to update.
    index : int
        The index of the feature to update.
    ub : float
        The new upper bound for the feature.
    lb : float
        The new lower bound for the feature.
    """
    rule.lbs[index] = max(rule.lbs[index], lb)
    rule.ubs[index] = min(rule.ubs[index], ub)

@boundscheck(False)
@wraparound(False)
cdef inline bint check_rule(Rule* rule, vector[int] feature_indices) :
    """
    Checks whether the rule has been fully defined for the given feature indices.

    Parameters
    ----------
    rule : Rule*
        The Rule struct to check.
    feature_indices : int[:]
        A list of feature indices to check.

    Returns
    -------
    bint
        True if all feature bounds have been specified (i.e., not infinite), False otherwise.
    """
    cdef int idx
    for idx in feature_indices:
        if (rule.lbs[idx] == -INFINITY) and (rule.ubs[idx] == INFINITY):
            return 0
    return 1

@boundscheck(False)
@wraparound(False)
cdef inline bint check_value(Rule* rule, int i, double value) noexcept nogil:
    """
    Checks if the given value lies within the bounds for the specified feature index.

    Parameters
    ----------
    rule : Rule*
        The Rule struct to check.
    i : int
        The feature index to check.
    value : double
        The value to check against the feature's bounds.

    Returns
    -------
    bint
        True if the value lies within the bounds, False otherwise.
    """
    return rule.lbs[i] < value <= rule.ubs[i]

cdef class Explainer2:
    cdef vector[vector[Rule]] trees
    cdef object model
    cdef int len_col
    cdef list columns

    def __init__(self):
        self.trees = vector[vector[Rule]]()
        self.model = None
        self.len_col = -1
        self.columns = None

    def __call__(self, model):
        self.model = model
        self.columns = model.feature_name()
        self.len_col = len(self.columns)
        self._analyze_tree()


    cdef void _analyze_tree(self):
        cdef object trees_df, tree
        cdef object root_node
        cdef int tree_index
        cdef str root_index 
        
        cdef Rule rule
        cdef vector[vector[Rule]] trees
        cdef vector[Rule] temp_rules, rules = vector[Rule]()
        cdef cnp.ndarray[cnp.int64_t, ndim=1] tree_indexs

        trees_df = self.model.trees_to_dataframe()
        tree_indexs = trees_df.tree_index.unique()
        

        for tree_index in tree_indexs:
            tree = trees_df[trees_df["tree_index"] == tree_index]
            root_node = tree[tree["parent_index"].isna()].iloc[0]
            root_index = root_node["node_index"]

            _traverse_tree(tree, root_index, {}, rules, tree_index, self.len_col)

        for tree_index in tree_indexs:
            temp_rules = vector[Rule]()
            for rule in rules:
                if rule.tree_index == tree_index:
                    temp_rules.push_back(rule)

            sort(temp_rules.begin(), temp_rules.end(), compare_rules)

            self.trees.push_back(temp_rules)

    @boundscheck(False)
    @wraparound(False)
    cdef vector[vector[Rule]] _filter_trees(
        self, int main_col, int sub_col = -1):
        """
        Filters rules across all trees based on the specified main and optional sub-column.

        Parameters
        ----------
        main_col : int
            The primary column index to filter by.
        sub_col : Optional[int], optional
            The secondary column index to filter by, by default None.

        Returns
        -------
        list
            A list of lists, where each inner list contains rules that match the given conditions.
        """

        cdef Rule rule
        cdef vector[vector[Rule]] filtered_trees
        cdef vector[Rule] filtered_rules = vector[Rule]()
        cdef vector[Rule] tree
        cdef vector[int] check_rules = vector[int]()

        # Convert Python int to C++ int
        cdef int c_main_col = <int>main_col
        cdef int c_sub_col = <int>sub_col
        
        check_rules.push_back(c_main_col)

        if c_sub_col != -1:
            check_rules.push_back(c_sub_col)

        for tree in self.trees:
            filtered_rules.clear()
            for rule in tree:
                if check_rule(&rule, check_rules):
                    filtered_rules.push_back(rule)

            if filtered_rules.size() >= 1:
                filtered_trees.push_back(filtered_rules)

        return filtered_trees


    @boundscheck(False)
    @wraparound(False)
    cpdef vector[double] _get_split_point(self, vector[vector[Rule]] trees, int col):
        """
        Retrieves the unique split points for a specific column across all trees, sorted in ascending order.

        Parameters
        ----------
        trees : vector[vector[Rule]]
            The list of trees, where each tree is a vector of Rule objects.
        col : int
            The column index for which to retrieve split points.

        Returns
        -------
        vector[double]
            A vector of unique split points for the specified column, excluding the smallest one.
        """

        cdef vector[double] points
        cdef vector[Rule] tree
        cdef Rule rule
        cdef vector[double].iterator it
        cdef vector[double] result

        for tree in trees:
            for rule in tree:
                points.push_back(rule.ubs[col])
                points.push_back(rule.lbs[col])

        sort(points.begin(), points.end())
        it = unique(points.begin(), points.end())
        
        points.resize(it - points.begin())

        if len(points) > 1:
            points.erase(points.begin())
            
        return points


    @boundscheck(False)
    @wraparound(False)
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
        cdef vector[vector[Rule]] filtered_trees = self._filter_trees(main_col, sub_col)
        cdef vector[double] main_split_points = self._get_split_point(filtered_trees, main_col)
        cdef vector[double] sub_split_points = self._get_split_point(filtered_trees, sub_col)

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
        cdef list results = []
        
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

                    mean_values.push_back(_find_mean(all_values))
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

        df = _replace_inf(df, main_column_name)
        df = _replace_inf(df, sub_column_name)

        return df
