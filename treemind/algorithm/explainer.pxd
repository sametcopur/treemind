from libcpp.vector cimport vector
from .rule cimport Rule

cimport numpy as cnp

cdef class Result:
    cdef dict _data
    cdef public int degree
    cdef public int n_classes
    cdef public list feature_names
    cdef public str model_type

    cdef _reorder_dataframes(self, dict class_data, tuple stored_key, tuple requested_key)
    cdef _reorder_single_dataframe(self, object df, tuple stored_key, tuple requested_key, list reorder_mapping)
    cdef _combine_class_dataframes(self, dict class_data)

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

    cpdef object count_node(self, int degree=?)

    cdef object prepare_dataframe(self, vector[int] col_indices, 
                    int num_cols, 
                    list column_names,
                    vector[vector[float]]& points,
                    vector[float]& mean_values,
                    vector[float]& ensemble_std,
                    vector[float]& counts,
                    object columns)