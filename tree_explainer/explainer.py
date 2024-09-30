import pandas as pd
import numpy as np
from fractions import Fraction
from typing import List, Tuple, Optional, Union


def _find_mean(tree_results: List[List[float]]) -> float:
    """
    Calculates the average of the sums of combinations in a memory-efficient way for large inputs.

    Parameters
    ----------
    tree_results : List[List[float]]
        A list of lists where each inner list contains float values representing possible
        outcomes from a tree. The function calculates the average over all possible combinations
        of one element from each list.

    Returns
    -------
    average : float
        The average of the sums of all possible combinations of one element from each sublist.
    """
    sums = [sum(sublist) for sublist in tree_results]
    element_counts = [len(sublist) for sublist in tree_results]

    combination_count = 1
    for element_count in element_counts:
        combination_count *= element_count

    total_sum = Fraction(0)
    for sum_value, element_count in zip(sums, element_counts):
        total_sum += Fraction(sum_value) * (combination_count // element_count)

    return float(total_sum / combination_count)


def _find_min_max(
    tree_results: List[List[float]],
) -> Tuple[float, float]:
    """
    Finds the minimum and maximum possible values from tree combinations.

    Parameters
    ----------
    tree_results : List[List[float]]
        A list of lists where each inner list represents possible outcomes from a tree.
        Each inner list contains float values corresponding to the outputs of that tree.

    Returns
    -------
    min_value : float
        The minimum possible value obtained by summing the minimum value from each list.

    max_value : float
        The maximum possible value obtained by summing the maximum value from each list.
    """
    min_value = 0
    max_value = 0

    for tree_output in tree_results:
        min_value += min(tree_output)
        max_value += max(tree_output)

    return min_value, max_value


def _replace_inf(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    unique_points = data[column_name].unique()
    unique_points = np.sort(unique_points[np.logical_not(np.isinf(unique_points))])
    max_main = unique_points.max()
    difference_main = max_main - (
        unique_points[-2] if unique_points.size > 1 else max_main
    )

    data.loc[np.isinf(data[column_name]), column_name] = max_main + difference_main

    return data


class Rule:
    """
    A class to represent and update decision rules for a given leaf in a tree.

    Parameters
    ----------
    len_col : int
        The number of columns (features) in the data.
    tree_index : int
        The index of the tree in the ensemble.
    leaf_index : int
        The index of the leaf in the tree.

    Attributes
    ----------
    lbs : np.ndarray
        Lower bounds for each feature, initialized to -infinity.
    ubs : np.ndarray
        Upper bounds for each feature, initialized to infinity.
    value : Optional[float]
        The predicted value associated with this rule, initially set to None.
    tree_index : int
        The index of the tree this rule belongs to.
    leaf_index : int
        The index of the leaf this rule corresponds to.
    """

    def __init__(self, len_col: int, tree_index: int, leaf_index: int) -> None:
        self.len_col = len_col
        self.tree_index = tree_index
        self.leaf_index = leaf_index
        self.lbs = np.full(shape=len_col, fill_value=-np.inf, dtype=np.float64)
        self.ubs = np.full(shape=len_col, fill_value=np.inf, dtype=np.float64)
        self._value: Optional[float] = None

    @property
    def value(self) -> Optional[float]:
        """
        Gets the predicted value for this rule.

        Returns
        -------
        value : Optional[float]
            The predicted value for this rule.
        """
        return self._value

    @value.setter
    def value(self, new_value: float) -> None:
        """
        Sets the predicted value for this rule.

        Parameters
        ----------
        new_value : float
            The new predicted value to set.
        """
        self._value = new_value

    def update_rule(
        self, index: int, ub: Optional[float] = None, lb: Optional[float] = None
    ) -> None:
        """
        Updates the lower and upper bounds for a feature at the specified index.

        Parameters
        ----------
        index : int
            The index of the feature to update.
        ub : Optional[float], optional
            The new upper bound for the feature, by default None.
        lb : Optional[float], optional
            The new lower bound for the feature, by default None.
        """
        if lb is not None:
            self.lbs[index] = max(self.lbs[index], lb)
        if ub is not None:
            self.ubs[index] = min(self.ubs[index], ub)

    def check_rule(self, i: list[int]) -> bool:
        """
        Checks whether the rule has been fully defined for the given feature indices.

        Parameters
        ----------
        i : list[int]
            A list of feature indices to check.

        Returns
        -------
        bool
            True if all feature bounds have been specified, False otherwise.
        """
        return np.all(
            np.logical_or(
                np.logical_not(np.isinf(self.lbs[i])),
                np.logical_not(np.isinf(self.ubs[i])),
            )
        )

    def check_value(self, i: int, value: float) -> bool:
        """
        Checks if the given value lies within the bounds for the specified feature index.

        Parameters
        ----------
        i : int
            The feature index to check.
        value : float
            The value to check against the feature's bounds.

        Returns
        -------
        bool
            True if the value lies within the bounds, False otherwise.
        """
        return self.lbs[i] < value <= self.ubs[i]

    def __repr__(self) -> str:
        """
        Returns a string representation of the rule, displaying non-infinite bounds,
        the predicted value, the leaf index, and the tree index.

        Returns
        -------
        str
            A formatted string representing the rule.
        """
        bounds = []
        for i in range(self.len_col):
            if not (np.isinf(self.lbs[i]) and np.isinf(self.ubs[i])):
                bounds.append(f"Feature_{i}: [{self.lbs[i]:.3f}, {self.ubs[i]:.3f}]")
        bounds_str = ", ".join(bounds) if bounds else "No bounds specified"
        return f"Rule({bounds_str}, Value: {self.value}, Leaf: {self.leaf_index}, Tree: {self.tree_index})"


class Explainer:
    def __init__(self):
        self.trees = []
        self.model = None
        self.len_col = None
        self.columns = None

    def __call__(self, model):
        self.model = model
        self.columns = model.feature_name()
        self.len_col = len(self.columns)
        self._analyze_tree()

        return self

    def _analyze_tree(self):

        trees_df = self.model.trees_to_dataframe()
        tree_indexs = trees_df.tree_index.unique()
        rules = []

        for tree_index in tree_indexs:
            tree = trees_df[trees_df["tree_index"] == tree_index]

            def traverse_tree(node_index, feature_ranges):
                node = tree[tree["node_index"] == node_index].iloc[0]

                if pd.isna(node["split_feature"]):
                    node_index = node["node_index"]
                    node_index = int(node_index[node_index.find("L") + 1 :])
                    rule = Rule(self.len_col, tree_index, node_index)
                    for feature_index, (lb, ub) in feature_ranges.items():
                        rule.update_rule(feature_index, lb=lb, ub=ub)
                    rule.value = node["value"]
                    rules.append(rule)
                    return

                feature_index = int(node["split_feature"].replace("Column_", ""))
                threshold = node["threshold"]

                left_ranges = feature_ranges.copy()
                left_ranges[feature_index] = (
                    left_ranges.get(feature_index, (float("-inf"), float("inf")))[0],
                    threshold,
                )

                right_ranges = feature_ranges.copy()
                right_ranges[feature_index] = (
                    threshold,
                    right_ranges.get(feature_index, (float("-inf"), float("inf")))[1],
                )

                traverse_tree(node["left_child"], left_ranges)
                traverse_tree(node["right_child"], right_ranges)

            root_node = tree[tree["parent_index"].isna()].iloc[0]
            root_index = root_node["node_index"]

            traverse_tree(root_index, {})

        for tree_index in tree_indexs:
            temp_rules = []
            for rule in rules:
                if rule.tree_index == tree_index:
                    temp_rules.append(rule)

            temp_rules = sorted(temp_rules, key=lambda x: x.leaf_index)
            self.trees.append(temp_rules)

    def _filter_trees(
        self, main_col: int, sub_col: Optional[int] = None
    ) -> List[List[float]]:
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
        return [
            [
                rule
                for rule in tree
                if rule.check_rule([main_col])
                and (sub_col is None or rule.check_rule([sub_col]))
            ]
            for tree in self.trees
        ]

    def _get_split_point(self, trees: List[List[Rule]], col: int) -> np.ndarray:
        """
        Retrieves the unique split points for a specific column across all trees, sorted in ascending order.

        Parameters
        ----------
        trees : List[List[Rule]]
            The list of trees, where each tree is a list of Rule objects.
        col : int
            The column index for which to retrieve split points.

        Returns
        -------
        np.ndarray
            An array of unique split points for the specified column, excluding the smallest one.
        """

        points = []
        for tree in trees:
            for rule in tree:
                points.append(rule.ubs[col])
                points.append(rule.lbs[col])

        unique_points = np.sort(np.unique(points))

        return unique_points[1:] if len(unique_points) > 1 else unique_points

    def analyze_row(
        self, x: np.ndarray, detailed: bool = True
    ) -> Union[Tuple[np.ndarray, List[np.ndarray], float], Tuple[np.ndarray, float]]:
        """
        Analyzes a single row of input data to extract predicted values and split points.

        Parameters
        ----------
        x : np.ndarray
            Input data for which predictions are made.
        detailed : bool, optional
            If True, returns detailed split points for each feature. Default is True.

        Returns
        -------
        Union[Tuple[np.ndarray, List[np.ndarray], float], Tuple[np.ndarray, float]]
            If detailed is True, returns a tuple of values array, list of split points, and raw score.
            Otherwise, returns a tuple of values array and raw score.
        """

        leafs = self.model.predict(x, pred_leaf=True)
        raw_score = np.mean(self.model.predict(x, raw_score=True))

        if detailed:
            split_points_list = [
                self._get_split_point(self.trees, col) for col in range(self.len_col)
            ]
            max_len = max(len(points) for points in split_points_list)
            values = np.full((self.len_col, max_len), fill_value=0.0, dtype=np.float64)
        else:
            values = np.full((self.len_col,), fill_value=0.0, dtype=np.float64)

        for row in range(leafs.shape[0]):
            for tree, leaf in zip(self.trees, leafs[row, :]):
                rule = tree[leaf]
                for col in range(self.len_col):
                    ub, lb = rule.ubs[col], rule.lbs[col]

                    if np.isinf(ub) and np.isinf(lb):
                        continue

                    if detailed:
                        loc_split_points = split_points_list[col]

                        values[col, : len(loc_split_points)] += np.where(
                            (lb < loc_split_points) & (loc_split_points <= ub),
                            rule.value,
                            0,
                        )
                    else:
                        values[col] += rule.value

        np.divide(values, leafs.shape[0], out=values)

        return (
            (values, split_points_list, raw_score) if detailed else (values, raw_score)
        )

    def analyze_dependency(self, main_col: int, sub_col: int) -> pd.DataFrame:
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
        pd.DataFrame
            A DataFrame with sub feature split points, main feature split points, and the corresponding values.
        """
        filtered_trees = self._filter_trees(main_col, sub_col)

        main_split_points = self._get_split_point(filtered_trees, main_col)
        sub_split_points = self._get_split_point(filtered_trees, sub_col)

        results = []
        for sub_point in sub_split_points:
            for main_point in main_split_points:

                all_values = [
                    [
                        rule.value
                        for rule in tree
                        if rule.check_value(main_col, main_point)
                        & rule.check_value(sub_col, sub_point)
                    ]
                    for tree in filtered_trees
                ]

                all_values = [vals for vals in all_values if vals]

                if not all_values:
                    continue

                results.append((sub_point, main_point, _find_mean(all_values)))

        main_column_name = self.columns[main_col]
        sub_column_name = self.columns[sub_col]

        df = pd.DataFrame(
            results, columns=[sub_column_name, main_column_name, "values"]
        )
        df = df.explode(["values"]).reset_index(drop=True)

        df = _replace_inf(df, main_column_name)
        df = _replace_inf(df, sub_column_name)

        return df

    def analyze_feature(self, col: int) -> pd.DataFrame:
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
        filtered_trees = self._filter_trees(col, None)

        split_points = self._get_split_point(filtered_trees, col).tolist()

        results = []
        for main_point in split_points:
            all_values = [
                [rule.value for rule in tree if rule.check_value(col, main_point)]
                for tree in filtered_trees
            ]

            all_values = [vals for vals in all_values if vals]

            if not all_values:
                continue

            min_val, max_val = _find_min_max(all_values)
            results.append((main_point, _find_mean(all_values), min_val, max_val))

        column_name = self.columns[col]
        df = pd.DataFrame(results, columns=[column_name, "mean", "min", "max"])
        df = _replace_inf(df, column_name)

        return df
