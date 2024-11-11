from libcpp.vector cimport vector
from .rule cimport Rule

cimport numpy as cnp

ctypedef unsigned long size_t

cdef double cmax(double a, double b) noexcept nogil

cdef double cmin(double a, double b) noexcept nogil

cdef object replace_inf(object data, str column_name)

cdef tuple _analyze_feature(int col, const vector[vector[Rule]] trees)

cdef tuple[vector[double], vector[double], vector[double],vector[double], vector[double]] _analyze_interaction(const vector[vector[Rule]] trees, int main_col, int sub_col)

cdef double _expected_value(int col, const vector[vector[Rule]] trees)

cdef add_lower_bound(object data, int loc, str column)

cdef tuple[vector[vector[double]],
           vector[double],
           vector[double],
           vector[double]] _analyze_multi_interaction(const vector[vector[Rule]]& trees, const vector[int]& columns)