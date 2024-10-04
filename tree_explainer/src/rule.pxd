from libcpp.vector cimport vector

cdef struct Rule:
    int len_col
    int tree_index
    int leaf_index
    vector[double] lbs 
    vector[double] ubs 
    double value   

cdef Rule create_rule(int len_col, int tree_index, int leaf_index)

cdef void update_rule(Rule* rule, int index, double lb, double ub)

cdef bint check_rule(Rule* rule, vector[int] feature_indices)

cdef bint check_value(Rule* rule, int i, double value) noexcept nogil

cdef int compare_rules(const Rule& a, const Rule& b)

cdef vector[double] get_split_point(vector[vector[Rule]] trees, int col)

cdef vector[vector[Rule]] filter_trees(vector[vector[Rule]] trees, int main_col, int sub_col= ?)