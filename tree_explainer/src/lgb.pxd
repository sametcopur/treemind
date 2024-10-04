
from libcpp.vector cimport vector
from .rule cimport Rule

cdef void traverse_lightgbm(object tree, str node_index, dict feature_ranges, vector[Rule]& rules, int tree_index, int len_col)

cdef vector[vector[Rule]] analyze_lightgbm( object model, int len_col)
  