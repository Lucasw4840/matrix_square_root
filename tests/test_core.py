import math
import unittest

from matrix_square_root import matrix_sqrt, MatrixSqrtError, is_matrix


class TestIsMatrix(unittest.TestCase):
    def test_valid_square(self):
        self.assertTrue(is_matrix([[1, 2], [3, 4]]))

    def test_valid_rectangular(self):
        self.assertTrue(is_matrix([[1, 2, 3], [4, 5, 6]]))

    def test_rejects_empty_outer(self):
        self.assertFalse(is_matrix([]))

    def test_rejects_empty_inner(self):
        self.assertFalse(is_matrix([[]]))

    def test_rejects_ragged(self):
        self.assertFalse(is_matrix([[1, 2], [3]]))

    def test_rejects_non_list_rows(self):
        self.assertFalse(is_matrix(["ab", "cd"]))


def _approx_equal(a, b, tol=1e-8):
    for i in range(len(a)):
        for j in range(len(a[0])):
            if abs(a[i][j] - b[i][j]) > tol:
                return False
    return True


class TestMatrixSqrt(unittest.TestCase):
    def test_identity(self):
        A = [[1, 0], [0, 1]]
        S = matrix_sqrt(A)
        self.assertTrue(_approx_equal(S, A, tol=1e-9))

    def test_diagonal(self):
        A = [[4, 0], [0, 9]]
        S = matrix_sqrt(A)
        self.assertTrue(_approx_equal(S, [[2, 0], [0, 3]], tol=1e-9))

    def test_positive_scalar_multiple_of_identity(self):
        A = [[16, 0, 0], [0, 16, 0], [0, 0, 16]]
        S = matrix_sqrt(A)
        expected = [[4, 0, 0], [0, 4, 0], [0, 0, 4]]
        self.assertTrue(_approx_equal(S, expected, tol=1e-9))

    def test_symmetric_positive_definite(self):
        # [[2,1],[1,2]] has eigenvalues 3 and 1, both positive -> principal sqrt exists.
        A = [[2, 1], [1, 2]]
        S = matrix_sqrt(A)
        product = [[
            S[0][0] * S[0][0] + S[0][1] * S[1][0],
            S[0][0] * S[0][1] + S[0][1] * S[1][1],
        ], [
            S[1][0] * S[0][0] + S[1][1] * S[1][0],
            S[1][0] * S[0][1] + S[1][1] * S[1][1],
        ]]
        self.assertTrue(_approx_equal(product, A, tol=1e-7))

    def test_upper_triangular(self):
        # [[9,4],[0,1]] -> sqrt is [[3, x],[0,1]] where 3*x + x*1 = 4, so x=1.
        A = [[9, 4], [0, 1]]
        S = matrix_sqrt(A)
        self.assertTrue(_approx_equal(S, [[3, 1], [0, 1]], tol=1e-7))

    def test_complex_eigenvalues(self):
        # Rotation by 90 degrees scaled by 4 has eigenvalues +-4i, which admit
        # a principal square root with positive-real eigenvalues +-2i... but the
        # iteration converges to the principal branch, whose eigenvalues are
        # those with non-negative real parts. We verify S@S == A, not the
        # specific eigenvalue split.
        A = [[0, -4], [4, 0]]
        S = matrix_sqrt(A)
        product = [[
            S[0][0] * S[0][0] + S[0][1] * S[1][0],
            S[0][0] * S[0][1] + S[0][1] * S[1][1],
        ], [
            S[1][0] * S[0][0] + S[1][1] * S[1][0],
            S[1][0] * S[0][1] + S[1][1] * S[1][1],
        ]]
        self.assertTrue(_approx_equal(product, A, tol=1e-6))

    def test_three_by_three(self):
        A = [[5, 1, 0], [1, 5, 1], [0, 1, 5]]
        S = matrix_sqrt(A)
        # Verify S @ S == A by manual product.
        n = 3
        product = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                product[i][j] = sum(S[i][k] * S[k][j] for k in range(n))
        self.assertTrue(_approx_equal(product, A, tol=1e-6))

    def test_returns_float_for_int_input(self):
        A = [[4, 0], [0, 9]]
        S = matrix_sqrt(A)
        # The iteration normalizes to float, so entries should be floats.
        self.assertTrue(all(isinstance(v, float) for row in S for v in row))

    def test_does_not_mutate_input(self):
        A = [[4, 0], [0, 9]]
        original = [row[:] for row in A]
        matrix_sqrt(A)
        self.assertEqual(A, original)

    def test_rejects_non_square(self):
        with self.assertRaises(MatrixSqrtError):
            matrix_sqrt([[1, 2, 3], [4, 5, 6]])

    def test_rejects_non_list(self):
        with self.assertRaises(MatrixSqrtError):
            matrix_sqrt("not a matrix")

    def test_rejects_empty(self):
        with self.assertRaises(MatrixSqrtError):
            matrix_sqrt([])
        with self.assertRaises(MatrixSqrtError):
            matrix_sqrt([[]])

    def test_singular_matrix_raises(self):
        # Zero matrix is singular; Denman-Beavers cannot invert Z_0 = I's
        # neighbor at the first step because Y_0 = 0 is singular.
        with self.assertRaises(MatrixSqrtError):
            matrix_sqrt([[0, 0], [0, 0]])

    def test_max_iter_zero_raises(self):
        # With max_iter=0 the loop never executes, so we hit the non-convergence
        # branch. This documents that max_iter is a hard cap, not a hint.
        with self.assertRaises(MatrixSqrtError):
            matrix_sqrt([[4, 0], [0, 9]], max_iter=0)

    def test_tol_is_relative(self):
        # A large matrix should converge with the default tol even though its
        # entries are huge; this verifies the relative-tolerance scaling.
        A = [[10_000, 0], [0, 90_000]]
        S = matrix_sqrt(A)
        self.assertTrue(_approx_equal(S, [[100, 0], [0, 300]], tol=1e-6))


if __name__ == "__main__":
    unittest.main()
