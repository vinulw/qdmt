import numpy as np
from ncon import ncon
from scipy.sparse.linalg import eigs, LinearOperator, gmres

def createMPS(D, d):
    """
    Returns a random complex MPS tensor.

        Parameters
        ----------
        D : int
            Bond dimension for MPS.
        d : int
            Physical dimension for MPS.

        Returns
        -------
        A : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            normalized.
    """

    A = np.random.rand(D, d, D) + 1j * np.random.rand(D, d, D)

    return normalizeMPS(A)


def createTransfermatrix(A):
    """
    Form the transfermatrix of an MPS.

        Parameters
        ----------
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.

        Returns
        -------
        E : np.array(D, D, D, D)
            Transfermatrix with 4 legs,
            ordered topLeft-bottomLeft-topRight-bottomRight.
    """

    E = ncon((A, np.conj(A)), ([-1, 1, -3], [-2, 1, -4]))

    return E


def normalizeMPS(A):
    """
    Normalize an MPS tensor.

        Parameters
        ----------
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.

        Returns
        -------
        Anew : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.

        Complexity
        ----------
        O(D ** 3) algorithm,
            D ** 3 contraction for transfer matrix handle.
    """

    D = A.shape[0]

    # calculate transfer matrix handle and cast to LinearOperator
    handleERight = lambda v: np.reshape(ncon((A, np.conj(A), v.reshape((D,D))), ([-1, 2, 1], [-2, 2, 3], [1, 3])),
                                        D ** 2)
    E = LinearOperator((D ** 2, D ** 2), matvec=handleERight)

    # calculate eigenvalue
    lam = eigs(E, k=1, which='LM', return_eigenvectors=False)

    Anew = A / np.sqrt(lam)

    return Anew


def leftFixedPoint(A):
    """
    Find left fixed point.

        Parameters
        ----------
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.

        Returns
        -------
        l : np.array(D, D)
            left fixed point with 2 legs,
            ordered bottom-top.

        Complexity
        ----------
        O(D ** 3) algorithm,
            D ** 3 contraction for transfer matrix handle.
    """

    D = A.shape[0]

    # calculate transfer matrix handle and cast to LinearOperator
    handleELeft = lambda v: np.reshape(ncon((A, np.conj(A), v.reshape((D, D))), ([1, 2, -2], [3, 2, -1], [3, 1])), D ** 2)
    E = LinearOperator((D ** 2, D ** 2), matvec=handleELeft)

    # calculate fixed point
    _, l = eigs(E, k=1, which='LM')

    # reshape to matrix
    l = l.reshape((D, D))

    # make left fixed point hermitian explicitly
    l /= (np.trace(l) / np.abs(np.trace(l)))# remove possible phase
    l = (l + np.conj(l).T) / 2 # force hermitian
    l *= np.sign(np.trace(l)) # force positive semidefinite

    return l


def rightFixedPoint(A):
    """
    Find right fixed point.

        Parameters
        ----------
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.

        Returns
        -------
        r : np.array(D, D)
            right fixed point with 2 legs,
            ordered top-bottom.

        Complexity
        ----------
        O(D ** 3) algorithm,
            D ** 3 contraction for transfer matrix handle.
    """

    D = A.shape[0]

    # calculate transfer matrix handle and cast to LinearOperator
    handleERight = lambda v: np.reshape(ncon((A, np.conj(A), v.reshape((D,D))), ([-1, 2, 1], [-2, 2, 3], [1, 3])), D ** 2)
    E = LinearOperator((D ** 2, D ** 2), matvec=handleERight)

    # calculate fixed point
    _, r = eigs(E, k=1, which='LM')

    # reshape to matrix
    r = r.reshape((D, D))

    # make right fixed point hermitian explicitly
    r /= (np.trace(r) / np.abs(np.trace(r)))# remove possible phase
    r = (r + np.conj(r).T) / 2 # force hermitian
    r *= np.sign(np.trace(r)) # force positive semidefinite

    return r


def fixedPoints(A):
    """
    Find normalized fixed points.

        Parameters
        ----------
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.

        Returns
        -------
        l : np.array(D, D)
            left fixed point with 2 legs,
            ordered bottom-top.
        r : np.array(D, D)
            right fixed point with 2 legs,
            ordered top-bottom.

        Complexity
        ----------
        O(D ** 3) algorithm,
            D ** 3 contraction for transfer matrix handle.
    """

    # find fixed points
    l, r = leftFixedPoint(A), rightFixedPoint(A)

    # calculate trace
    trace = np.trace(l@r)

    return l / trace, r
