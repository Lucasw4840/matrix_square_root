"""Core implementation of the principal matrix square root.

The principal square root of a matrix A is the unique matrix X with strictly
positive-real eigenvalues such that X @ X == A. The Denman-Beavers iteration is
an iteration scheme that converges quadratically to the principal square root.

A Python-only implementation of matrix operations is used here because the
library has a zero third-party dependency requirement. For small dense matrices
up to roughly 50x50, the pure-Python approach is fast enough on real-world input.
"""


import cmath
import math


__all__ = ["matrix_sqrt", "MatrixSqrtError", "is_matrix"]


class MatrixSqrtError(ValueError):
    """Raised when the principal square root does not exist or cannot be reached.

    Subclassing ValueError means callers can catch the broad category without
    importing this class; the more specific class exists for those who want to
    distinguish a malformed matrix from a non-converging iteration.
    """


def is_matrix(value):
    """Return True if value is a rectangular list-of-lists with at least one row.

    We treat a matrix as a list of lists where every row is a list of the same
    length. Empty rows are not allowed because their algebraic meaning is
    ambiguous (a 0-column matrix), and supporting them would complicate every
    downstream function for no real benefit.
    """
    if not isinstance(value, list) or not value:
        return False
    if not isinstance(value[0], list):
        return False
    n = len(value[0])
    if n == 0:
        return False
    for row in value:
        if not isinstance(row, list) or len(row) != n:
            return False
    return True


def _matmul(a, b):
    """Multiply two square matrices of the same dimension.

    Specialized to the square case because the Denman-Beavers iteration only
    ever multiplies square matrices of the same size; a general matmul would do
    more validation work per call than is needed.
    """
    n = len(a)
    m = len(b[0])
    p = len(b)
    result = [[0] * m for _ in range(n)]
    for i in range(n):
        ai = a[i]
        ri = result[i]
        for k in range(p):
            aik = ai[k]
            if aik == 0:
                continue
            bk = b[k]
            for j in range(m):
                ri[j] += aik * bk[j]
    return result


def _matadd(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def _matscale(s, a):
    return [[s * v for v in row] for row in a]


def _identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def _inverse(a):
    """Invert a square matrix via Gauss-Jordan elimination with partial pivoting.

    Partial pivoting (choosing the row with the largest |leading entry|) keeps
    numerical error bounded even when a column's leading entry happens to be
    small. Full pivoting would be marginally more stable but noticeably slower
    in pure Python, so we stop at partial pivoting as a deliberate trade-off.

    Complex entries are supported because the iteration may produce them when
    the input has complex eigenvalues; abs() handles both int and complex.
    """
    n = len(a)
    aug = [list(a[i]) + [1 if i == j else 0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        pivot_val = aug[pivot_row][col]
        if pivot_val == 0:
            raise MatrixSqrtError("Matrix is singular; cannot invert during iteration.")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        inv_pivot = 1 / pivot_val
        for j in range(2 * n):
            aug[col][j] *= inv_pivot
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0:
                continue
            row_r = aug[r]
            row_col = aug[col]
            for j in range(2 * n):
                row_r[j] -= factor * row_col[j]
    return [row[n:] for row in aug]


def _frobenius_norm(a):
    total = 0.0
    for row in a:
        for v in row:
            total += abs(v) ** 2
    return math.sqrt(total)


def _allclose(a, b, tol):
    if len(a) != len(b) or (a and len(a[0]) != len(b[0])):
        return False
    for i in range(len(a)):
        for j in range(len(a[0])):
            if abs(a[i][j] - b[i][j]) > tol:
                return False
    return True


def _entry_types(a):
    """Return the set of python type names appearing in the matrix.

    Used to decide whether the result of an integer-input iteration should be
    reported as floats or ints. Because Denman-Beavers always involves a
    matrix inverse, integer-only matrices almost never stay integral; we still
    check so callers who pass floats get floats back, not a surprise complex.
    """
    types = set()
    for row in a:
        for v in row:
            types.add(type(v).__name__)
    return types


def matrix_sqrt(matrix, *, tol=1e-12, max_iter=100):
    """Compute the principal square root of a dense square matrix.

    Uses the Denman-Beavers iteration:

        Y_0 = A,  Z_0 = I
        Y_{k+1} = 0.5 * (Y_k + Z_k^{-1})
        Z_{k+1} = 0.5 * (Z_k + Y_k^{-1})

    Y_k converges quadratically to A^{1/2}. Iteration stops when successive
    Y values agree within tol (relative to the Frobenius norm) or when
    max_iter is reached. Reaching max_iter without convergence raises
    MatrixSqrtError, which in practice means the matrix has no principal
    square root reachable from this iteration (e.g. a singular or defective
    matrix with eigenvalues on the negative real axis).

    Args:
        matrix: A square list-of-lists. Entries may be int, float, or complex.
        tol:   Convergence tolerance on successive iterates. Default 1e-12.
        max_iter: Hard cap on iteration count. Default 100.

    Returns:
        A new list-of-lists S with S @ S approximately equal to the input.

    Raises:
        MatrixSqrtError: on non-square / empty input, or on non-convergence.
    """
    if not is_matrix(matrix):
        raise MatrixSqrtError("Input must be a non-empty rectangular list-of-lists.")
    n = len(matrix)
    if n != len(matrix[0]):
        raise MatrixSqrtError("Matrix must be square; got {}x{}.".format(n, len(matrix[0])))

    # Copy so we never mutate caller data. We also normalize entries to float
    # (or complex if the input already contains complex); the iteration is
    # numerically unstable on raw ints because of repeated division.
    has_complex = "complex" in _entry_types(matrix)
    if has_complex:
        y = [[complex(v) for v in row] for row in matrix]
    else:
        y = [[float(v) for v in row] for row in matrix]
    z = _identity(n)
    if has_complex:
        z = [[complex(v) for v in row] for row in z]

    prev = None
    for _ in range(max_iter):
        y_inv = _inverse(y)
        z_inv = _inverse(z)
        y_new = _matscale(0.5, _matadd(y, z_inv))
        z_new = _matscale(0.5, _matadd(z, y_inv))
        if prev is not None:
            # Relative tolerance on the change in Y. Using the Frobenius norm
            # of the previous iterate as the denominator handles both tiny and
            # large matrices with the same tol value.
            denom = _frobenius_norm(prev)
            if denom == 0:
                denom = 1.0
            if _allclose(y, y_new, tol * denom):
                y = y_new
                z = z_new
                break
        prev = y
        y = y_new
        z = z_new
    else:
        raise MatrixSqrtError("Denman-Beavers iteration did not converge within {} iterations.".format(max_iter))

    return y
