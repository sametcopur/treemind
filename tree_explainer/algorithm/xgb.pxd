from libcpp.vector cimport vector
from .rule cimport Rule


cdef vector[vector[Rule]] analyze_xgboost(object model, int len_col)
cdef int[:,::1] xgb_leaf_correction(vector[vector[Rule]] trees, int[:,::1] leafs)
cdef object convert_d_matrix(object x)

