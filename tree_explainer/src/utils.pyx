from libcpp.vector cimport vector
from libcpp.algorithm cimport max_element, min_element

from cython cimport boundscheck, wraparound

import numpy as np
cimport numpy as cnp


@boundscheck(False)
@wraparound(False)
cdef inline double max(double a, double b):
    return a if a > b else b

@boundscheck(False)
@wraparound(False)
cdef inline double min(double a, double b):
    return a if a < b else b

cdef object replace_inf(object data, str column_name):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] unique_points_ = np.asarray(data[column_name].unique(), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] unique_points = np.sort(unique_points_[np.logical_not(np.isinf(unique_points_))])
    cdef double max_main = unique_points.max()
    cdef double difference_main = max_main - (unique_points[-2] if unique_points.size > 1 else max_main)

    data.loc[np.isinf(data[column_name]), column_name] = max_main + difference_main

    return data

@boundscheck(False)
@wraparound(False)
cdef double find_mean(vector[vector[double]] tree_results) noexcept nogil:
    """
    Calculates the average of the sums of combinations in a memory-efficient way for large inputs.

    Parameters
    ----------
    tree_results : vector[vector[float]]
        A vector of vectors where each inner vector contains float values representing possible
        outcomes from a tree. The function calculates the average over all possible combinations
        of one element from each vector.

    Returns
    -------
    average : float
        The average of the sums of all possible combinations of one element from each sublist.
    """
    cdef double sums = 0.0
    cdef vector[double] sublist
    cdef double sublist_mean
    cdef double total
    cdef int i
    cdef int sublist_size 

    for sublist in tree_results:
        sublist_size = sublist.size()
        total = 0.0
        for i in range(sublist_size):
            total += sublist[i]
        
        sums += total / sublist_size
    
    return sums

@boundscheck(False)
@wraparound(False)
cdef tuple[double, double] find_min_max(vector[vector[double]] tree_results) noexcept nogil:
    """
    Finds the minimum and maximum possible values from tree combinations using C++ vectors.

    Parameters
    ----------
    tree_results : vector[vector[double]]
        A C++ vector of vectors where each inner vector represents possible outcomes from a tree.
        Each inner vector contains float values corresponding to the outputs of that tree.

    Returns
    -------
    min_value : double
        The minimum possible value obtained by summing the minimum value from each vector.

    max_value : double
        The maximum possible value obtained by summing the maximum value from each vector.
    """
    cdef double min_value = 0.0
    cdef double max_value = 0.0
    cdef int i
    cdef vector[double] vector_i

    for i in range(tree_results.size()):
        vector_i = tree_results[i]

        if vector_i.size() > 1:
            min_value += <double>min_element(vector_i.begin(), vector_i.end())[0]
            max_value += <double>max_element(vector_i.begin(), vector_i.end())[0]
        else:
            min_value += vector_i[0]
            max_value += vector_i[0]

    return min_value, max_value