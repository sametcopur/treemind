from libcpp.vector cimport vector

cdef struct Rule:
    double value
    int count
    vector[double] lbs 
    vector[double] ubs 
    int len_col
    int tree_index
    int leaf_index



cdef Rule create_rule(int len_col, int tree_index, int leaf_index)

cdef void update_rule(Rule* rule, int index, double lb, double ub)

cdef bint check_rule(const Rule* rule, vector[int] feature_indices)  noexcept nogil

cdef bint check_value(const Rule* rule, int i, double value) noexcept nogil

cdef int compare_rules(const Rule& a, const Rule& b)

cdef vector[double] get_split_point(vector[vector[Rule]] trees, int col)

cdef vector[vector[Rule]] filter_trees(vector[vector[Rule]] trees, int main_col, int sub_col= ?)