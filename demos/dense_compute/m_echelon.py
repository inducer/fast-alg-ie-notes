import numpy as np


def swap_rows(mat, i, j):
    temp = mat[i].copy()
    mat[i] = mat[j]
    mat[j] = temp


def m_echelon(A):
    """Return a tuple (M, echelon) so that *echelon* is in upper echelon form,
    *M* is invertible, and ``M.dot(A)`` (approximately) equals *echelon*.
    """

    m, n = A.shape
    M = np.eye(m)
    U = A.copy()

    row = 0
    for col in range(n):
        piv_row = row + np.argmax(np.abs(U[row:, col]))

        if abs(U[piv_row, col]) == 0:
            # column is exactly zero
            continue

        swap_rows(U, row, piv_row)
        swap_rows(M, row, piv_row)

        for el_row in range(row + 1, m):
            fac = -U[el_row, col]/U[row, col]
            U[el_row] += fac*U[row]
            M[el_row] += fac*M[row]

        row += 1

        if row + 1 >= m:
            break

    return M, U


