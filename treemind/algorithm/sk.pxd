from libcpp.vector cimport vector
from .rule cimport Rule

cdef tuple[vector[vector[Rule]], vector[vector[int]], vector[int]] analyze_sklearn(object model, int len_col, int n_classes)