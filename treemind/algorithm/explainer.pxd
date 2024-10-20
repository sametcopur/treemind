from libcpp.vector cimport vector
from .rule cimport Rule

cdef class Explainer:
    cdef vector[vector[Rule]] trees
    cdef public object model
    cdef int len_col
    cdef list columns
    cdef str model_type

    cpdef object analyze_interaction(self, int main_col, int sub_col)
    cpdef tuple analyze_data(self, object x, bint detailed = ?)
    cpdef object analyze_feature(self, int col)
    cpdef object count_node(self, bint interaction=?)