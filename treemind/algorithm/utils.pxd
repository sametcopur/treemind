from libcpp.vector cimport vector
from .rule cimport Rule

cimport numpy as cnp

ctypedef unsigned long size_t

cdef float cmax(float a, float b) noexcept nogil

cdef float cmin(float a, float b) noexcept nogil

cdef object replace_inf(object data, str column_name)

cdef add_lower_bound(object data, int loc, str column)

cdef tuple[vector[vector[float]],
           vector[float],
           vector[float],
           vector[float]] _analyze_feature(const vector[vector[Rule]]& trees,
                                            const vector[int]& columns,
                                            const vector[vector[int]]& cat_cols)

cdef vector[vector[Rule]] filter_class_trees(const vector[vector[Rule]]& trees,  
                                              const int n_classes, 
                                              const int class_idx)