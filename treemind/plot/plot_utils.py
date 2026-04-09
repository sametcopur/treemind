import pandas as pd
import numpy as np

# ---- Visual constants ----
COLOR_PRIMARY = "#3A5894"
COLOR_DISTRIBUTION = "#4C72B0"
COLOR_STD = "gray"

HSPACE_STACKED = -0.1
EXTEND_RATIO = 0.05
INFINITY_FALLBACK_FRACTION = 0.1


def _classify_features(df, meta_cols=frozenset({"value", "std", "count", "class"})):
    """
    Identify features in a DataFrame and classify as 'continuous' or 'categorical'.

    Returns list of (feature_name, feature_type, column_tuple) in column order.
    Continuous features have <name>_lb and <name>_ub columns; all others are categorical.
    """
    features = []
    visited = set()

    for col in df.columns:
        if col in meta_cols or col in visited:
            continue

        if col.endswith("_lb"):
            base = col[:-3]
            ub_col = f"{base}_ub"
            if ub_col in df.columns:
                features.append((base, "continuous", (col, ub_col)))
                visited.update((col, ub_col))
                continue

        if col.endswith("_ub"):
            base = col[:-3]
            lb_col = f"{base}_lb"
            if lb_col in df.columns:
                # Already handled via _lb branch
                continue

        features.append((col, "categorical", (col,)))
        visited.add(col)

    return features


def _extract_feature_name(df, meta_cols=frozenset({"value", "std", "count", "class"})):
    """Extract a human-readable feature name from a result DataFrame."""
    features = _classify_features(df, meta_cols)
    if not features:
        return "Feature"
    return features[0][0]


def _find_tick_decimal(ticks: np.ndarray, n_ticks: int) -> int:
    tick_min, tick_max = np.min(ticks), np.max(ticks)
    tick_range = tick_max - tick_min

    if tick_range < n_ticks:
        for decimals_count in range(1, 5):
            rounded_ticks = np.round(ticks, decimals_count)
            if np.allclose(ticks, rounded_ticks, atol=tick_range / (n_ticks * 10)):
                return decimals_count
        return 4
    else:
        return 0


def _replace_infinity(
    data: pd.DataFrame, column_name: str, infinity_type: str = "positive"
):
    """
    Replaces positive or negative infinity values in a DataFrame column with a calculated value
    based on the finite values in the column.

    Parameters:
    - data: pd.DataFrame
        The DataFrame containing the data.
    - column_name: str
        The name of the column to process.
    - infinity_type: str, optional (default='positive')
        Specify 'positive' to replace positive infinity values,
        'negative' to replace negative infinity values.

    Returns:
    - pd.DataFrame
        A copy of the DataFrame with infinity values replaced.
    """
    data = data.copy()
    # Extract unique finite values
    unique_values = np.asarray(data[column_name].unique(), dtype=np.float64)
    finite_values = unique_values[~np.isinf(unique_values)]
    finite_values_sorted = np.sort(finite_values)

    if infinity_type == "positive":
        # Handle positive infinity
        max_value = finite_values_sorted.max()
        if finite_values_sorted.size > 1:
            second_max = finite_values_sorted[-2]
            difference = max_value - second_max
        else:
            difference = max_value * INFINITY_FALLBACK_FRACTION
        # Replace positive infinity values
        data.loc[np.isposinf(data[column_name]), column_name] = np.float32(max_value + difference)

    elif infinity_type == "negative":
        # Handle negative infinity
        min_value = finite_values_sorted.min()
        if finite_values_sorted.size > 1:
            second_min = finite_values_sorted[1]
            difference = second_min - min_value
        else:
            difference = abs(min_value) * INFINITY_FALLBACK_FRACTION
        # Replace negative infinity values
        data.loc[np.isneginf(data[column_name]), column_name] = np.float32(min_value - difference)

    else:
        raise ValueError("infinity_type must be 'positive' or 'negative'")

    return data
