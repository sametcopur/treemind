from libcpp.vector cimport vector
from .rule cimport Rule

cimport numpy as cnp

cdef class Explainer:
    cdef vector[vector[Rule]] trees
    cdef vector[vector[int]] cat_cols
    cdef vector[int] cat_indices
    cdef public object model
    cdef object categorical
    cdef Py_ssize_t len_col
    cdef list columns
    cdef str model_type
    cdef bint must_backdata

    cpdef object count_node(self, int order=?)