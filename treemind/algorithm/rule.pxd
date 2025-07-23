from libcpp.vector cimport vector
cimport numpy as cnp

cdef struct Rule:
    float value
    float count
    vector[float] lbs 
    vector[float] ubs
    vector[vector[bint]] cats 
    int len_col
    int tree_index
    int leaf_index
    vector[bint] cat_flags


cdef Rule create_rule(int len_col, int tree_index, int leaf_index)

cdef void update_cats_for_rule(Rule* rule, const vector[vector[bint]]& cats)

cdef void update_rule(Rule* rule, int index, float lb, float ub)

cdef bint check_rule(const Rule* rule, vector[int] feature_indices)  noexcept nogil

cdef bint check_value(const Rule* rule, int i, float value) noexcept nogil

cdef int compare_rules(const Rule& a, const Rule& b)

cdef vector[float] get_split_point(vector[vector[Rule]] trees, int col)

cdef vector[vector[Rule]] update_leaf_counts(vector[vector[Rule]] trees, object model, object back_data, str model_type)