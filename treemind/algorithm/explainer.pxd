from libcpp.vector cimport vector

cimport numpy as cnp

cdef struct Rule:
    float value
    float count
    vector[float] lbs
    vector[float] ubs
    vector[vector[bint]] cats
    int len_col
    int tree_index
    int leaf_index
    vector[bint] cat_flags

cdef class Result:
    cdef dict _data
    cdef public int degree
    cdef public int n_classes
    cdef public list feature_names
    cdef public str model_type
    cdef vector[int] cat_indices
    cdef object categorical

    cdef _build_dataframe(self, dict class_data, tuple feature_key)

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