from libcpp.vector cimport vector
from .rule cimport Rule

cimport numpy as cnp

cdef class Explainer:
    cdef vector[vector[Rule]] trees
    cdef public object model
    cdef int len_col
    cdef list columns
    cdef str model_type

    cpdef object analyze_interaction(self, int main_col, int sub_col)
    cpdef cnp.ndarray[cnp.float64_t, ndim=2] analyze_data(self, object x, object back_data = ?)
    cpdef object analyze_feature(self, int col)
    cpdef object count_node(self, bint interaction=?)