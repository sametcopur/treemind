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
    cdef int i

    rule.len_col = len_col
    rule.tree_index = tree_index
    rule.leaf_index = leaf_index
    rule.lbs = vector[float](len_col, -INFINITY)
    rule.ubs = vector[float](len_col, INFINITY)
    rule.value = np.nan
    rule.count = -1
    rule.cats = vector[vector[bint]]()
    rule.cat_flags = vector[bint]()
    return rule

cdef void update_cats_for_rule(Rule* rule, const vector[vector[bint]]& cats):
    """
    Copies the active categorical mask constraints into the Rule struct and
    prints them (debug).

    Parameters
    ----------
    rule : Rule*
        The Rule struct being constructed for a leaf node.
    cats : vector[vector[bint]]&
        Temporary categorical mask collected during tree traversal.
        Each cats[i] is a boolean mask indicating allowed category values for feature i.
    """
    cdef size_t i, j
    cdef bint all_one
    rule.cats.resize(cats.size())
    rule.cat_flags.resize(cats.size(), 0)

    for i in range(cats.size()):
        rule.cats[i] = cats[i]

        all_one = True
        for j in range(cats[i].size()):
            if cats[i][j] == 0:
                all_one = False
                break
        rule.cat_flags[i] = 0 if all_one else 1

cdef void update_rule(Rule* rule, int index, float lb, float ub):
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


cdef inline bint check_rule(const Rule* rule, vector[int] feature_indices)  noexcept nogil:
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

cdef inline bint check_value(const Rule* rule, int i, float value) noexcept nogil:
    """
    Checks if the given value lies within the bounds for the specified feature index.

    Parameters
    ----------
    rule : Rule*
        The Rule struct to check.
    i : int
        The feature index to check.
    value : float
        The value to check against the feature's bounds.

    Returns
    -------
    bint
        True if the value lies within the bounds, False otherwise.
    """
    return rule.lbs[i] < value <= rule.ubs[i]



cdef int compare_rules(const Rule& a, const Rule& b):
    return a.leaf_index < b.leaf_index


cdef vector[float] get_split_point(vector[vector[Rule]] trees, int col):
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
    vector[float]
        A vector of unique split points for the specified column, excluding the smallest one.
    """

    cdef vector[float] points
    cdef vector[Rule] tree
    cdef Rule rule
    cdef vector[float].iterator it
    cdef vector[float] result

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
