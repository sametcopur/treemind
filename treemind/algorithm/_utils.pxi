cdef vector[vector[Rule]] filter_class_trees(const vector[vector[Rule]]& trees,
                                              const int n_classes,
                                              const int class_idx) noexcept nogil:
    cdef vector[vector[Rule]] class_trees = vector[vector[Rule]]()
    cdef size_t i
    cdef int tree_idx

    # Number of trees per class = total trees / number of classes
    class_trees.resize(trees.size() // n_classes)

    for i in range(class_trees.size()):
        tree_idx = i * n_classes + class_idx
        class_trees[i] = trees[tree_idx]

    return class_trees


cdef inline void _add_box_3d(
    vector[float]& arr,
    size_t dim1,
    size_t dim2,
    size_t x0,
    size_t x1,
    size_t y0,
    size_t y1,
    size_t z0,
    size_t z1,
    float value
) noexcept nogil:
    arr[(x0 * dim1 + y0) * dim2 + z0] += value
    arr[(x1 * dim1 + y0) * dim2 + z0] -= value
    arr[(x0 * dim1 + y1) * dim2 + z0] -= value
    arr[(x0 * dim1 + y0) * dim2 + z1] -= value
    arr[(x1 * dim1 + y1) * dim2 + z0] += value
    arr[(x1 * dim1 + y0) * dim2 + z1] += value
    arr[(x0 * dim1 + y1) * dim2 + z1] += value
    arr[(x1 * dim1 + y1) * dim2 + z1] -= value


cdef inline void _prefix_sum_3d(
    vector[float]& arr,
    size_t size0,
    size_t size1,
    size_t size2
) noexcept nogil:
    cdef size_t x, y, z, idx, base, prev_base
    cdef size_t dim1 = size1 + 1
    cdef size_t dim2 = size2 + 1
    cdef size_t plane = dim1 * dim2

    for x in range(size0 + 1):
        for y in range(size1 + 1):
            base = (x * dim1 + y) * dim2
            for z in range(1, size2 + 1):
                arr[base + z] += arr[base + z - 1]

    for x in range(size0 + 1):
        for y in range(1, size1 + 1):
            base = (x * dim1 + y) * dim2
            prev_base = base - dim2
            for z in range(size2 + 1):
                arr[base + z] += arr[prev_base + z]

    for x in range(1, size0 + 1):
        base = x * plane
        prev_base = base - plane
        for idx in range(plane):
            arr[base + idx] += arr[prev_base + idx]


cdef tuple[vector[vector[float]],
        vector[float],
        vector[float],
        vector[float]] _analyze_feature(const vector[vector[Rule]]& trees,
                                                const vector[int]& columns,
                                                const vector[vector[int]]& cat_cols):
    cdef:
        vector[vector[float]] split_points_list = vector[vector[float]]()
        size_t num_cols = columns.size()
        size_t max_size = 1
        size_t num_trees = trees.size()
        size_t i, j, k, idx, tree_idx, rule_idx

        # Statistics vectors
        vector[float] average_counts
        vector[float] mean_values
        vector[float] ensemble_std

        # Per-tree temporary storage
        vector[float] tree_count
        vector[float] tree_sum
        vector[float] tree_squared_sum

        vector[vector[size_t]] valid_indices_per_col = vector[vector[size_t]](num_cols)
        vector[size_t] multi_index = vector[size_t](num_cols)
        vector[size_t] sizes = vector[size_t](num_cols)

        float rule_value, rule_count, squared_value
        float mean, variance, std, count
        bint skip_rule
        const Rule* rule_ptr
        const vector[Rule]* tree_ptr

        cdef bint done

        vector[vector[float]] combined_points
        size_t col_idx, remaining, point_idx, split_size
        int col_k
        float val
        int val_idx
        vector[vector[int]] cat_slice = vector[vector[int]]()

        vector[bint] is_categorical = vector[bint](num_cols)

        size_t total_split_points = 0

    # Pre-compute categorical flags and prepare data
    for i in range(columns.size()):
        col_idx = columns[i]
        cat_slice.push_back(cat_cols[col_idx])
        is_categorical[i] = (cat_slice[i].size() > 0)

    for col_idx in range(num_cols):
        if not is_categorical[col_idx]:  # NUMERICAL
            split_points = get_split_point(trees, columns[col_idx])
        else:  # CATEGORICAL
            split_points = vector[float]()
            split_points.reserve(cat_slice[col_idx].size())  # Only for categorical
            for i in range(cat_slice[col_idx].size()):
                split_points.push_back(cat_slice[col_idx][i])

        split_points_list.push_back(split_points)
        sizes[col_idx] = split_points.size()
        total_split_points += split_points.size()
        max_size *= sizes[col_idx]

        valid_indices_per_col[col_idx].reserve(split_points.size())

    # Initialize result vectors
    average_counts = vector[float](max_size, 0.0)
    mean_values = vector[float](max_size, 0.0)
    ensemble_std = vector[float](max_size, 0.0)

    # Initialize temporary vectors
    tree_count = vector[float](max_size)
    tree_sum = vector[float](max_size)
    tree_squared_sum = vector[float](max_size)

    with nogil:
        # Process each tree
        for tree_idx in range(num_trees):
            tree_ptr = &trees[tree_idx]

            # Reset per-tree accumulators (more efficient than assign)
            for idx in range(max_size):
                tree_sum[idx] = 0.0
                tree_squared_sum[idx] = 0.0
                tree_count[idx] = 0.0

            # Process each rule in the current tree
            for rule_idx in range(tree_ptr.size()):
                rule_ptr = &(tree_ptr[0][rule_idx])
                rule_value = rule_ptr.value
                rule_count = rule_ptr.count
                squared_value = rule_value * rule_value

                skip_rule = False

                for k in range(num_cols):
                    valid_indices_per_col[k].clear()
                    split_size = split_points_list[k].size()
                    col_k = columns[k]  # cache vector lookup

                    # Adaptive capacity - only grow if needed
                    if valid_indices_per_col[k].capacity() < split_size:
                        valid_indices_per_col[k].reserve(split_size)

                    if not is_categorical[k]:  # NUMERICAL - optimized branch
                        for i in range(split_size):
                            val = split_points_list[k][i]
                            if check_value(rule_ptr, col_k, val):
                                valid_indices_per_col[k].push_back(i)
                    else:  # CATEGORICAL - optimized branch
                        for i in range(split_size):
                            val_idx = <int>split_points_list[k][i]
                            if rule_ptr.cats[col_k][val_idx]:
                                valid_indices_per_col[k].push_back(i)

                    if valid_indices_per_col[k].empty():
                        skip_rule = True
                        break

                if skip_rule:
                    continue

                if num_cols == 1:
                    # Special case for single column - avoid multi-index overhead
                    for i in range(valid_indices_per_col[0].size()):
                        idx = valid_indices_per_col[0][i]
                        tree_sum[idx] += rule_value * rule_count
                        tree_squared_sum[idx] += squared_value * rule_count
                        tree_count[idx] += rule_count
                elif num_cols == 2:
                    # Special case for two columns - common case optimization
                    for i in range(valid_indices_per_col[0].size()):
                        for j in range(valid_indices_per_col[1].size()):
                            idx = valid_indices_per_col[0][i] * sizes[1] + valid_indices_per_col[1][j]
                            tree_sum[idx] += rule_value * rule_count
                            tree_squared_sum[idx] += squared_value * rule_count
                            tree_count[idx] += rule_count

                elif num_cols == 3:
                    # Optimized 3-column case - nested loops
                    for i in range(valid_indices_per_col[0].size()):
                        for j in range(valid_indices_per_col[1].size()):
                            for k in range(valid_indices_per_col[2].size()):
                                idx = (valid_indices_per_col[0][i] * sizes[1] + valid_indices_per_col[1][j]) * sizes[2] + valid_indices_per_col[2][k]

                                tree_sum[idx] += rule_value * rule_count
                                tree_squared_sum[idx] += squared_value * rule_count
                                tree_count[idx] += rule_count

                elif num_cols == 4:
                    # Optimized 4-column case
                    for i in range(valid_indices_per_col[0].size()):
                        for j in range(valid_indices_per_col[1].size()):
                            for k in range(valid_indices_per_col[2].size()):
                                for l in range(valid_indices_per_col[3].size()):
                                    idx = ((valid_indices_per_col[0][i] * sizes[1] + valid_indices_per_col[1][j]) * sizes[2] + valid_indices_per_col[2][k]) * sizes[3] + valid_indices_per_col[3][l]

                                    tree_sum[idx] += rule_value * rule_count
                                    tree_squared_sum[idx] += squared_value * rule_count
                                    tree_count[idx] += rule_count

                else:
                    # General case - optimized multi-index iteration
                    multi_index.assign(num_cols, 0)
                    done = False
                    while not done:
                        idx = valid_indices_per_col[0][multi_index[0]]
                        for k in range(1, num_cols):
                            idx = idx * sizes[k] + valid_indices_per_col[k][multi_index[k]]

                        # Update statistics
                        tree_sum[idx] += rule_value * rule_count
                        tree_squared_sum[idx] += squared_value * rule_count
                        tree_count[idx] += rule_count

                        k = num_cols - 1
                        while k >= 0:
                            multi_index[k] += 1
                            if multi_index[k] < valid_indices_per_col[k].size():
                                break
                            multi_index[k] = 0
                            k -= 1
                        if k < 0:
                            done = True

            for idx in range(max_size):
                count = tree_count[idx]
                if count > 0:
                    average_counts[idx] += count
                    mean = tree_sum[idx] / count

                    # More numerically stable variance calculation
                    variance = (tree_squared_sum[idx] - tree_sum[idx] * mean) / count
                    if variance < 0.0:
                        variance = 0.0
                    std = sqrt(variance)

                    ensemble_std[idx] += std
                    mean_values[idx] += mean

        # Finalize aggregate statistics
        for idx in range(max_size):
            average_counts[idx] /= num_trees

    combined_points = vector[vector[float]](max_size)
    for idx in range(max_size):
        combined_points[idx] = vector[float](num_cols)
        remaining = idx
        for k in range(num_cols - 1, -1, -1):
            point_idx = remaining % sizes[k]
            remaining //= sizes[k]
            combined_points[idx][k] = split_points_list[k][point_idx]

    return combined_points, mean_values, ensemble_std, average_counts


cdef void _analyze_feature_nogil(
    const vector[vector[Rule]]& trees,
    const vector[int]& columns,
    const vector[vector[int]]& cat_cols,
    const vector[vector[float]]& split_cache,
    vector[vector[float]]& out_points,
    vector[float]& out_means,
    vector[float]& out_stds,
    vector[float]& out_counts
) noexcept nogil:
    """Fully nogil version of _analyze_feature with cached split points."""
    cdef:
        vector[vector[float]] split_points_list = vector[vector[float]]()
        size_t num_cols = columns.size()
        size_t max_size = 1
        size_t num_trees = trees.size()
        size_t i, j, k, l, idx, tree_idx, rule_idx
        size_t idx0, idx1, idx2, idx3

        vector[float] average_counts
        vector[float] mean_values
        vector[float] ensemble_std

        vector[float] tree_count
        vector[float] tree_sum
        vector[float] tree_squared_sum
        vector[float] diff_count
        vector[float] diff_sum
        vector[float] diff_squared_sum

        vector[vector[size_t]] valid_indices_per_col = vector[vector[size_t]](num_cols)
        vector[size_t] multi_index = vector[size_t](num_cols)
        vector[size_t] sizes = vector[size_t](num_cols)
        vector[size_t] active_sizes = vector[size_t](num_cols)
        vector[size_t] range_starts = vector[size_t](num_cols)
        vector[bint] use_numeric_ranges = vector[bint](num_cols)

        float rule_value, rule_count, squared_value
        float mean, variance, std, count
        bint skip_rule
        const Rule* rule_ptr
        const vector[Rule]* tree_ptr
        bint done

        size_t col_idx, remaining, point_idx, split_size
        size_t range_end
        size_t diff_size, dim1_padded, dim2_padded, flat_idx
        int col_k
        int val_idx
        Py_ssize_t k_rev
        vector[vector[int]] cat_slice = vector[vector[int]]()
        vector[bint] is_categorical = vector[bint](num_cols)
        vector[float] split_points
        bint use_numeric_3d_fast

    # Pre-compute categorical flags and split points
    for i in range(num_cols):
        col_idx = columns[i]
        cat_slice.push_back(cat_cols[col_idx])
        is_categorical[i] = (cat_slice[i].size() > 0)

    for col_idx in range(num_cols):
        if not is_categorical[col_idx]:
            split_points_list.push_back(split_cache[columns[col_idx]])
        else:
            split_points = vector[float]()
            split_points.reserve(cat_slice[col_idx].size())
            for i in range(cat_slice[col_idx].size()):
                split_points.push_back(cat_slice[col_idx][i])
            split_points_list.push_back(split_points)

        sizes[col_idx] = split_points_list[col_idx].size()
        max_size *= sizes[col_idx]
        valid_indices_per_col[col_idx].reserve(sizes[col_idx])

    use_numeric_3d_fast = (num_cols == 3)
    if use_numeric_3d_fast:
        for k in range(num_cols):
            if is_categorical[k]:
                use_numeric_3d_fast = 0
                break

    # Initialize statistics vectors
    average_counts = vector[float](max_size, 0.0)
    mean_values = vector[float](max_size, 0.0)
    ensemble_std = vector[float](max_size, 0.0)
    if use_numeric_3d_fast:
        dim1_padded = sizes[1] + 1
        dim2_padded = sizes[2] + 1
        diff_size = (sizes[0] + 1) * dim1_padded * dim2_padded
        diff_count = vector[float](diff_size)
        diff_sum = vector[float](diff_size)
        diff_squared_sum = vector[float](diff_size)
    else:
        tree_count = vector[float](max_size)
        tree_sum = vector[float](max_size)
        tree_squared_sum = vector[float](max_size)

    # Process each tree
    for tree_idx in range(num_trees):
        tree_ptr = &trees[tree_idx]

        if use_numeric_3d_fast:
            for idx in range(diff_size):
                diff_count[idx] = 0.0
                diff_sum[idx] = 0.0
                diff_squared_sum[idx] = 0.0
        else:
            for idx in range(max_size):
                tree_sum[idx] = 0.0
                tree_squared_sum[idx] = 0.0
                tree_count[idx] = 0.0

        for rule_idx in range(tree_ptr.size()):
            rule_ptr = &(tree_ptr[0][rule_idx])
            rule_value = rule_ptr.value
            rule_count = rule_ptr.count
            squared_value = rule_value * rule_value

            skip_rule = False

            for k in range(num_cols):
                col_k = columns[k]

                if not is_categorical[k]:
                    range_starts[k] = _upper_bound_float(split_points_list[k], rule_ptr.lbs[col_k])
                    range_end = _upper_bound_float(split_points_list[k], rule_ptr.ubs[col_k])
                    active_sizes[k] = range_end - range_starts[k]
                    use_numeric_ranges[k] = 1
                else:
                    valid_indices_per_col[k].clear()
                    split_size = split_points_list[k].size()

                    if valid_indices_per_col[k].capacity() < split_size:
                        valid_indices_per_col[k].reserve(split_size)

                    for i in range(split_size):
                        val_idx = <int>split_points_list[k][i]
                        if rule_ptr.cats[col_k][val_idx]:
                            valid_indices_per_col[k].push_back(i)
                    active_sizes[k] = valid_indices_per_col[k].size()
                    use_numeric_ranges[k] = 0

                if active_sizes[k] == 0:
                    skip_rule = True
                    break

            if skip_rule:
                continue

            if use_numeric_3d_fast:
                idx0 = range_starts[0]
                idx1 = idx0 + active_sizes[0]
                idx2 = range_starts[1]
                idx3 = idx2 + active_sizes[1]
                _add_box_3d(
                    diff_count, dim1_padded, dim2_padded,
                    idx0, idx1, idx2, idx3,
                    range_starts[2], range_starts[2] + active_sizes[2],
                    rule_count
                )
                _add_box_3d(
                    diff_sum, dim1_padded, dim2_padded,
                    idx0, idx1, idx2, idx3,
                    range_starts[2], range_starts[2] + active_sizes[2],
                    rule_value * rule_count
                )
                _add_box_3d(
                    diff_squared_sum, dim1_padded, dim2_padded,
                    idx0, idx1, idx2, idx3,
                    range_starts[2], range_starts[2] + active_sizes[2],
                    squared_value * rule_count
                )
            elif num_cols == 1:
                for i in range(active_sizes[0]):
                    if use_numeric_ranges[0]:
                        idx = range_starts[0] + i
                    else:
                        idx = valid_indices_per_col[0][i]
                    tree_sum[idx] += rule_value * rule_count
                    tree_squared_sum[idx] += squared_value * rule_count
                    tree_count[idx] += rule_count
            elif num_cols == 2:
                for i in range(active_sizes[0]):
                    if use_numeric_ranges[0]:
                        idx0 = range_starts[0] + i
                    else:
                        idx0 = valid_indices_per_col[0][i]

                    for j in range(active_sizes[1]):
                        if use_numeric_ranges[1]:
                            idx1 = range_starts[1] + j
                        else:
                            idx1 = valid_indices_per_col[1][j]

                        idx = idx0 * sizes[1] + idx1
                        tree_sum[idx] += rule_value * rule_count
                        tree_squared_sum[idx] += squared_value * rule_count
                        tree_count[idx] += rule_count
            elif num_cols == 3:
                for i in range(active_sizes[0]):
                    if use_numeric_ranges[0]:
                        idx0 = range_starts[0] + i
                    else:
                        idx0 = valid_indices_per_col[0][i]

                    for j in range(active_sizes[1]):
                        if use_numeric_ranges[1]:
                            idx1 = range_starts[1] + j
                        else:
                            idx1 = valid_indices_per_col[1][j]

                        for k in range(active_sizes[2]):
                            if use_numeric_ranges[2]:
                                idx2 = range_starts[2] + k
                            else:
                                idx2 = valid_indices_per_col[2][k]

                            idx = (idx0 * sizes[1] + idx1) * sizes[2] + idx2
                            tree_sum[idx] += rule_value * rule_count
                            tree_squared_sum[idx] += squared_value * rule_count
                            tree_count[idx] += rule_count
            elif num_cols == 4:
                for i in range(active_sizes[0]):
                    if use_numeric_ranges[0]:
                        idx0 = range_starts[0] + i
                    else:
                        idx0 = valid_indices_per_col[0][i]

                    for j in range(active_sizes[1]):
                        if use_numeric_ranges[1]:
                            idx1 = range_starts[1] + j
                        else:
                            idx1 = valid_indices_per_col[1][j]

                        for k in range(active_sizes[2]):
                            if use_numeric_ranges[2]:
                                idx2 = range_starts[2] + k
                            else:
                                idx2 = valid_indices_per_col[2][k]

                            for l in range(active_sizes[3]):
                                if use_numeric_ranges[3]:
                                    idx3 = range_starts[3] + l
                                else:
                                    idx3 = valid_indices_per_col[3][l]

                                idx = ((idx0 * sizes[1] + idx1) * sizes[2] + idx2) * sizes[3] + idx3
                                tree_sum[idx] += rule_value * rule_count
                                tree_squared_sum[idx] += squared_value * rule_count
                                tree_count[idx] += rule_count
            else:
                multi_index.assign(num_cols, 0)
                done = False
                while not done:
                    if use_numeric_ranges[0]:
                        idx = range_starts[0] + multi_index[0]
                    else:
                        idx = valid_indices_per_col[0][multi_index[0]]

                    for k in range(1, num_cols):
                        if use_numeric_ranges[k]:
                            point_idx = range_starts[k] + multi_index[k]
                        else:
                            point_idx = valid_indices_per_col[k][multi_index[k]]
                        idx = idx * sizes[k] + point_idx

                    tree_sum[idx] += rule_value * rule_count
                    tree_squared_sum[idx] += squared_value * rule_count
                    tree_count[idx] += rule_count

                    k_rev = <Py_ssize_t>num_cols - 1
                    while k_rev >= 0:
                        multi_index[k_rev] += 1
                        if multi_index[k_rev] < active_sizes[k_rev]:
                            break
                        multi_index[k_rev] = 0
                        k_rev -= 1
                    if k_rev < 0:
                        done = True

        if use_numeric_3d_fast:
            _prefix_sum_3d(diff_count, sizes[0], sizes[1], sizes[2])
            _prefix_sum_3d(diff_sum, sizes[0], sizes[1], sizes[2])
            _prefix_sum_3d(diff_squared_sum, sizes[0], sizes[1], sizes[2])

            for i in range(sizes[0]):
                for j in range(sizes[1]):
                    flat_idx = (i * dim1_padded + j) * dim2_padded
                    idx = (i * sizes[1] + j) * sizes[2]
                    for k in range(sizes[2]):
                        count = diff_count[flat_idx + k]
                        if count > 0:
                            average_counts[idx + k] += count
                            mean = diff_sum[flat_idx + k] / count
                            variance = (diff_squared_sum[flat_idx + k] - diff_sum[flat_idx + k] * mean) / count
                            if variance < 0.0:
                                variance = 0.0
                            std = sqrt(variance)
                            ensemble_std[idx + k] += std
                            mean_values[idx + k] += mean
        else:
            for idx in range(max_size):
                count = tree_count[idx]
                if count > 0:
                    average_counts[idx] += count
                    mean = tree_sum[idx] / count
                    variance = (tree_squared_sum[idx] - tree_sum[idx] * mean) / count
                    if variance < 0.0:
                        variance = 0.0
                    std = sqrt(variance)
                    ensemble_std[idx] += std
                    mean_values[idx] += mean

    for idx in range(max_size):
        average_counts[idx] /= num_trees

    # Build combined points and write to output
    out_points = vector[vector[float]](max_size)
    for idx in range(max_size):
        out_points[idx] = vector[float](num_cols)
        remaining = idx
        k_rev = <Py_ssize_t>num_cols - 1
        while k_rev >= 0:
            point_idx = remaining % sizes[k_rev]
            remaining //= sizes[k_rev]
            out_points[idx][k_rev] = split_points_list[k_rev][point_idx]
            k_rev -= 1

    out_means = mean_values
    out_stds = ensemble_std
    out_counts = average_counts
