import pandas as pd
import numpy as np
from fractions import Fraction

def _efficient_solution(lists):
    """Calculates the average of the sums of combinations in a memory-efficient way for large numbers."""
    
    # Calculate the sum and length of each sublist
    sums = [sum(sublist) for sublist in lists]
    element_counts = [len(sublist) for sublist in lists]
    
    # Calculate the total number of combinations
    combination_count = 1
    for element_count in element_counts:
        combination_count *= element_count
    
    # Find the contributions to the total for calculating the average
    total_sum = Fraction(0)
    for sum_value, element_count in zip(sums, element_counts):
        # Each sublist's sum repeats for the number of elements in the other lists
        total_sum += Fraction(sum_value) * (combination_count // element_count)
    
    # Calculate the average
    average = total_sum / combination_count
    
    return float(average)


def _min_max_combinations_efficient(lists):
    """Efficiently finds the minimum and maximum possible values from combinations."""
    min_value = sum(min(sublist) for sublist in lists)  # Minimum element from each sublist
    max_value = sum(max(sublist) for sublist in lists)  # Maximum element from each sublist
    
    return min_value, max_value



class Rule:
    def __init__(
        self, len_col: int, tree_index: int, leaf_index: int, frequency: float
    ) -> None:
        self.len_col = len_col
        self.tree_index = tree_index
        self.leaf_index = leaf_index
        self.frequency = frequency
        self.lbs = np.full(shape=len_col, fill_value=-np.inf, dtype=np.float64)
        self.ubs = np.full(shape=len_col, fill_value=np.inf, dtype=np.float64)
        self.value = None

    def update_rule(self, index, ub=None, lb=None):
        if lb is not None:
            self.lbs[index] = max(self.lbs[index], lb)
        if ub is not None:
            self.ubs[index] = min(self.ubs[index], ub)

    def update_value(self, value):
        self.value = value

    def check_rule(self, i: list[int]):
        return np.all(
            np.logical_or(
                np.logical_not(np.isinf(self.lbs[i])),
                np.logical_not(np.isinf(self.ubs[i])),
            )
        )

    def check_value(self, i: int, value: float):
        return self.lbs[i] < value <= self.ubs[i]

    def __repr__(self):
        bounds = []
        for i in range(self.len_col):
            if not (np.isinf(self.lbs[i]) and np.isinf(self.ubs[i])):
                bounds.append(f"Feature_{i}: [{self.lbs[i]}, {self.ubs[i]}]")
        bounds_str = ", ".join(bounds)
        return f"Rule({bounds_str}, Value: {self.value}, Leaf: {self.leaf_index}, Tree: {self.tree_index})"


class Tree:
    def __init__(self, model, X_train):
        self.trees = []
        self.model = model
        self.X_train = X_train
        self.len_col = X_train.shape[1]

    def _find_frequency(self):
        leafs = self.model.predict(self.X_train, pred_leaf=True)

        counts = []

        for i in range(leafs.shape[1]):
            _, count = np.unique(leafs[:, i], return_counts=True)
            counts.append(count / count.max())

        return counts

    def analyze_tree(self):
        trees_df = self.model.trees_to_dataframe()
        tree_indexs = trees_df.tree_index.unique()
        freequency_list = self._find_frequency()
        rules = []

        for tree_index in tree_indexs:
            tree = trees_df[trees_df["tree_index"] == tree_index]

            def traverse_tree(node_index, feature_ranges):
                node = tree[tree["node_index"] == node_index].iloc[0]

                if pd.isna(node["split_feature"]):
                    node_index = node["node_index"]
                    node_index = int(node_index[node_index.find("L") + 1 :])
                    freequency = freequency_list[tree_index][node_index]
                    rule = Rule(self.len_col, tree_index, node_index, freequency)
                    for feature_index, (lb, ub) in feature_ranges.items():
                        rule.update_rule(feature_index, lb=lb, ub=ub)
                    rule.update_value(node["value"])
                    rules.append(rule)
                    return

                feature_index = int(node["split_feature"].replace("Column_", ""))
                threshold = node["threshold"]

                # Sol çocuk için aralığı güncelle
                left_ranges = feature_ranges.copy()
                left_ranges[feature_index] = (
                    left_ranges.get(feature_index, (float("-inf"), float("inf")))[0],
                    threshold,
                )

                # Sağ çocuk için aralığı güncelle
                right_ranges = feature_ranges.copy()
                right_ranges[feature_index] = (
                    threshold,
                    right_ranges.get(feature_index, (float("-inf"), float("inf")))[1],
                )

                traverse_tree(node["left_child"], left_ranges)
                traverse_tree(node["right_child"], right_ranges)

            # Kök düğümü bul ve ağacı travers et
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

    def _filter_rules(self, main_col, sub_col=None):
        rules = []
        for tree in self.trees:
            temp_rules = []
            for rule in tree:
                if sub_col is None:
                    if rule.check_rule(main_col):
                        temp_rules.append(rule)

                else:
                    if rule.check_rule(main_col) & rule.check_rule(sub_col):
                        temp_rules.append(rule)

            rules.append(temp_rules)

        return rules

    def _get_split_point(self, filtered_rules, col):

        points = []
        for tree in filtered_rules:
            for rule in tree:
                points.append(rule.ubs[col])
                points.append(rule.lbs[col])

        return np.sort(np.unique(points))[1:]

    def analyze_row(self, x, detailed=True):

        leafs = self.model.predict(x, pred_leaf=True)
        raw_score = self.model.predict(x, raw_score=True)[0]

        if detailed:
            split_points_list = []
            max_len = 0
            for col in range(self.len_col):
                split_point = self._get_split_point(self.trees, col)
                split_points_list.append(split_point)
                split_point_len = len(split_point)

                if split_point_len > max_len:
                    max_len = split_point_len

            values = np.zeros((self.len_col, max_len), dtype=np.float64)

        else:
            values = np.zeros((self.len_col,), dtype=np.float64)

        for tree, leaf in zip(self.trees, leafs[0]):
            rule = tree[leaf]
            for col in range(self.len_col):
                ub = rule.ubs[col]
                lb = rule.lbs[col]
                if not np.isinf(ub) or not np.isinf(lb):
                    if detailed:
                        loc_split_points = split_points_list[col]

                        values[col, : len(loc_split_points)] += np.where(
                            (lb < loc_split_points) & (loc_split_points <= ub),
                            rule.value,
                            0,
                        )
                    else:
                        values[col] += rule.value

        if detailed:
            return values, split_points_list, raw_score
        else:
            return values, raw_score

    def analyze_dependency(self, main_col, sub_col):
        filtered_rules = self._filter_rules(main_col, sub_col)

        main_split_points = self._get_split_point(filtered_rules, main_col)

        sub_split_points = self._get_split_point(filtered_rules, sub_col)

        results = []
        for sub_point in sub_split_points:
            for main_point in main_split_points:
                all_values = []
                for tree in filtered_rules:
                    tree_values = []
                    for rule in tree:
                        if rule.check_value(main_col, main_point) & rule.check_value(
                            sub_col, sub_point
                        ):

                            tree_values.append(rule.value)

                    if tree_values:
                        all_values.append(tree_values)
                results.append((sub_point, main_point, _efficient_solution(all_values)))

        df = pd.DataFrame(results, columns=["sub_point", "main_point", "values"])

        df = df.explode(["values"]).reset_index(drop=True)
        df["values"] = df["values"].astype(float)

        # "main_point" için inf değerlerini güncelle
        non_inf_values_main = df["main_point"][df["main_point"] != float("inf")]
        sorted_main_values = non_inf_values_main.sort_values()
        max_main = sorted_main_values.max()
        previous_main = (
            sorted_main_values.iloc[-2] if len(sorted_main_values) > 1 else max_main
        )  # İkinci en büyük değer
        difference_main = max_main - previous_main

        df.loc[df["main_point"] == float("inf"), "main_point"] = (
            max_main + difference_main
        )

        # "sub_point" için inf değerlerini güncelle
        non_inf_values_sub = df["sub_point"][df["sub_point"] != float("inf")]
        sorted_sub_values = non_inf_values_sub.sort_values()
        max_sub = sorted_sub_values.max()
        previous_sub = (
            sorted_sub_values.iloc[-2] if len(sorted_sub_values) > 1 else max_sub
        )  # İkinci en büyük değer
        difference_sub = max_sub - previous_sub

        df.loc[df["sub_point"] == float("inf"), "sub_point"] = max_sub + difference_sub

        return df

    def analyze_feature_v2(self, col):

        filtered_trees = self._filter_rules(col, None)

        main_split_points = self._get_split_point(filtered_trees, col).tolist()

        results = []
        for main_point in main_split_points:
            all_values = []
            for tree in filtered_trees:
                tree_values = []
                for rule in tree:
                    if rule.check_value(col, main_point):
                        tree_values.append(rule.value)

                if tree_values:
                    all_values.append(tree_values)
                    
            min_val, max_val = _min_max_combinations_efficient(all_values)
            results.append((main_point, _efficient_solution(all_values), min_val, max_val))

        df = pd.DataFrame(results, columns=["main_point", "mean", "min", "max"])

        non_inf_values = df["main_point"][df["main_point"] != float("inf")]
        df.loc[df["main_point"] == float("inf"), "main_point"] = (
            non_inf_values.max() + non_inf_values.std() + 1e-6
        )

        return df

    def analyze_feature(self, data, main_col):

        leafs = self.model.predict(data, pred_leaf=True)

        filtered_trees = self._filter_rules(main_col, None)

        for tree in filtered_trees:
            if not tree:
                continue

            tree_index = tree[0].tree_index
            filtered_leafs = np.array([rule.leaf_index for rule in tree])
            leafs[:, tree_index] = np.where(
                np.isin(leafs[:, tree_index], filtered_leafs), leafs[:, tree_index], -1
            )

        main_split_points = self._get_split_point(filtered_trees, main_col).tolist()

        for_main = list(zip([-np.inf] + main_split_points[:-1], main_split_points))

        rule_values = [[rule.value for rule in rule_list] for rule_list in self.trees]

        data_main = data[:, main_col]

        results = []
        for main_lb, main_ub in for_main:

            mask = (data_main > main_lb) & (data_main <= main_ub)
            filtered_leafs = leafs[mask]
            values = np.zeros(filtered_leafs.shape[0], dtype=np.float64)

            for i, row_leafs in enumerate(filtered_leafs):
                for tree_index, leaf_index in enumerate(row_leafs):
                    if leaf_index != -1:
                        values[i] += rule_values[tree_index][leaf_index]

            results.append((main_ub, np.asarray(values)))

        df = pd.DataFrame(results, columns=["main_point", "values"])

        df = df.explode(["values"]).reset_index(drop=True)
        df["values"] = df["values"].astype(float)

        non_inf_values = df["main_point"][df["main_point"] != float("inf")]
        df.loc[df["main_point"] == float("inf"), "main_point"] = (
            non_inf_values.max() + non_inf_values.std() + 1e-6
        )

        return df

    def analyze_dependency_data(self, data, main_col, sub_col):

        leafs = self.model.predict(data, pred_leaf=True)

        filtered_trees = self._filter_rules(main_col, sub_col)

        for tree in filtered_trees:
            if not tree:
                continue

            tree_index = tree[0].tree_index
            filtered_leafs = np.array([rule.leaf_index for rule in tree])
            leafs[:, tree_index] = np.where(
                np.isin(leafs[:, tree_index], filtered_leafs), leafs[:, tree_index], -1
            )

        main_split_points = self._get_split_point(filtered_trees, main_col).tolist()
        sub_split_points = self._get_split_point(filtered_trees, sub_col).tolist()

        for_main = list(zip([-np.inf] + main_split_points[:-1], main_split_points))
        for_sub = list(zip([-np.inf] + sub_split_points[:-1], sub_split_points))

        rule_values = [[rule.value for rule in rule_list] for rule_list in self.trees]

        data_main = data[:, main_col]
        data_sub = data[:, sub_col]

        results = []
        for sub_lb, sub_ub in for_sub:
            for main_lb, main_ub in for_main:

                mask = (
                    (data_main > main_lb)
                    & (data_main <= main_ub)
                    & (data_sub > sub_lb)
                    & (data_sub <= sub_ub)
                )
                filtered_leafs = leafs[mask]
                values = np.zeros(filtered_leafs.shape[0], dtype=np.float64)

                for i, row_leafs in enumerate(filtered_leafs):
                    for tree_index, leaf_index in enumerate(row_leafs):
                        if leaf_index != -1:
                            values[i] += rule_values[tree_index][leaf_index]

                results.append((sub_ub, main_ub, np.asarray(values)))

        df = pd.DataFrame(results, columns=["sub_point", "main_point", "values"])

        df = df.explode(["values"]).reset_index(drop=True)
        df["values"] = df["values"].astype(float)

        non_inf_values = df["main_point"][df["main_point"] != float("inf")]
        df.loc[df["main_point"] == float("inf"), "main_point"] = (
            non_inf_values.max() + non_inf_values.std() + 1e-6
        )

        non_inf_values = df["sub_point"][df["sub_point"] != float("inf")]

        df.loc[df["sub_point"] == float("inf"), "sub_point"] = (
            non_inf_values.max() + non_inf_values.std() + 1e-5
        )

        return df
