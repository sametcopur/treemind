from libcpp.vector cimport vector
from .rule cimport Rule

cimport numpy as cnp

cdef class Explainer:
    cdef vector[vector[Rule]] trees
    cdef public object model
    cdef Py_ssize_t len_col
    cdef list columns
    cdef str model_type

    cpdef object count_node(self, int order=?)
    cpdef object analyze_feature(self, object columns, object back_data = ?)