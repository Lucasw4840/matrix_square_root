Matrix Square Root computes the principal square root of a small dense matrix
using the Denman-Beavers iteration, with zero third-party dependencies.

Usage:

    from matrix_square_root import matrix_sqrt

    A = [[4, 0], [0, 9]]
    S = matrix_sqrt(A)
    # S == [[2.0, 0.0], [0.0, 3.0]]

`matrix_sqrt(matrix, *, tol=1e-12, max_iter=100)` returns a new list-of-lists
`S` such that `S @ S` is approximately equal to `matrix`. It raises
`MatrixSqrtError` (a subclass of `ValueError`) on malformed input or on
non-convergence. Also exported: `is_matrix(value)` for checking whether a
value is a non-empty rectangular list-of-lists.

Why this exists: the principal matrix square root appears in covariance
smoothing, fractional matrix powers, and Lie-group linearization. A full
implementation would call LAPACK's `sqrtm`; that is the right answer for large
or production workloads. The trade-off here is that this library has no
dependencies and runs anywhere standard Python does, at the cost of being
O(n^3) per iteration in pure Python — practical up to roughly 50x50.

Edge case to know about: the Denman-Beavers iteration requires inverting the
running iterates, so it cannot produce a square root for a singular matrix
(such as the zero matrix) and will raise `MatrixSqrtError`. Matrices whose
eigenvalues lie exactly on the negative real axis also have no principal square
root reachable from this iteration; the same error is raised. The library
does not attempt a Schur decomposition fallback.

Entries may be `int`, `float`, or `complex`; the result is always `float` or
`complex` because the iteration divides at every step.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

## Limitations

Values are coerced to floats, so very large integers lose precision. If you need
exact integer aggregates over a window, this is the wrong tool.

## Contributing

Issues and pull requests are welcome. Please keep the dependency list empty —
that constraint is the point of the project, not an oversight.

