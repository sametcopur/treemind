from libcpp.vector cimport vector
from .rule cimport Rule

cdef tuple[vector[vector[Rule]], vector[vector[int]], vector[int], int] analyze_catboost(object model, int len_col)