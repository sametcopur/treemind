from libcpp.vector cimport vector
from .rule cimport Rule

cdef class Explainer:
    cdef vector[vector[Rule]] trees
    cdef object model
    cdef int len_col
    cdef list columns

    cpdef object analyze_dependency(self, int main_col, int sub_col)
    cpdef tuple analyze_row(self, object x, bint detailed = ?)
    cpdef object analyze_feature(self, int col)