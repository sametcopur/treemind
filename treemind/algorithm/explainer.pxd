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
    cdef int n_classes

    cpdef object count_node(self, int order=?)

    cdef object prepare_dataframe(self, vector[int] col_indices, 
                    int num_cols, 
                    list column_names,
                    vector[vector[double]]& points,
                    vector[double]& mean_values,
                    vector[double]& ensemble_std,
                    vector[double]& counts,
                    object columns)