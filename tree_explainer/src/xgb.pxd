from libcpp.vector cimport vector
from .rule cimport Rule


cdef vector[vector[Rule]] analyze_xgboost(object model, int len_col)


