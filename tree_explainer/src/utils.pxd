from libcpp.vector cimport vector
from .rule cimport Rule

cimport numpy as cnp

ctypedef unsigned long size_t

cdef double cmax(double a, double b) noexcept nogil

cdef double cmin(double a, double b) noexcept nogil

cdef object replace_inf(object data, str column_name)

cdef tuple[vector[double], vector[double], vector[double], vector[double]] _analyze_feature(int col, vector[vector[Rule]] trees)

cdef tuple[vector[double], vector[double], vector[double]] _analyze_dependency(vector[vector[Rule]] trees, int main_col, int sub_col)
   
cdef vector[double] pre_allocate_vector(size_t size) noexcept nogil