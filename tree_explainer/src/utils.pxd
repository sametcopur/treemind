from libcpp.vector cimport vector

cdef double max(double a, double b)

cdef double min(double a, double b)

cdef object replace_inf(object data, str column_name)

cdef double find_mean(vector[vector[double]] tree_results) noexcept nogil
    
cdef tuple[double, double] find_min_max(vector[vector[double]] tree_results)  noexcept nogil