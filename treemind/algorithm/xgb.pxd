from libcpp.vector cimport vector
from .rule cimport Rule

cimport numpy as cnp


cdef tuple[vector[vector[Rule]], vector[vector[int]], vector[int], int]  analyze_xgboost(object model, int len_col)
cdef cnp.ndarray[cnp.int32_t, ndim=2, mode="c"] xgb_leaf_correction(vector[vector[Rule]] trees, int[:,:] leafs)
cdef object convert_d_matrix(object x)

