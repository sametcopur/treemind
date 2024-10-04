
from libcpp.vector cimport vector
from .rule cimport Rule
from libcpp.pair cimport pair as cpp_pair

ctypedef cpp_pair[double, double] RangePair

cdef void traverse_lightgbm(dict node_dict, str node_index, vector[RangePair]& feature_ranges, vector[Rule]& rules, int tree_index)
cdef vector[vector[Rule]] analyze_lightgbm(object model, int len_col)