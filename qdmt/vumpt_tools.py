'''
Tool methods to help with vumps tutorial

Source: VUMPs tutorial (https://github.com/leburgel/uniformMpsTutorial)
'''
from scipy.linalg import rq, qr, svd, polar
from scipy.sparse.linalg import eigs, LinearOperator, gmres
import numpy as np
from ncon import ncon

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


def rqPos(A):
    """
    Do a RQ decomposition with positive diagonal elements for R.
        Parameters
        ----------
        A : np.array(M, N)
            Matrix to decompose.
        Returns
        -------
        R : np.array(M, M)
            Upper triangular matrix,
            positive diagonal elements.
        Q : np.array(M, N)
            Orthogonal matrix.
        Complexity
        ----------
        ~O(max(M, N) ** 3) algorithm.
    """

    M, N = A.shape

    # LQ decomposition: scipy conventions: Q.shape = (N, N), L.shape = (M, N)
    R, Q = rq(A)

    # Throw out zeros under diagonal: Q.shape = (M, N), L.shape = (M, M)
    Q = Q[-M:, :]
    R = R[:, -M:]

    # Extract signs and multiply with signs on diagonal
    diagSigns = np.diag(np.sign(np.diag(R)))
    Q = np.dot(diagSigns, Q)
    R = np.dot(R, diagSigns)

    return R, Q


def rightOrthonormalize(A, R0=None, tol=1e-14, maxIter=1e5):
    """
    Transform A to right-orthonormal gauge.
        Parameters
        ----------
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.
        R0 : np.array(D, D), optional
            Right gauge matrix,
            initial guess.
        tol : float, optional
            convergence criterium,
            norm(R - Rnew) < tol.
        maxIter : int
            maximum amount of iterations.
        Returns
        -------
        R : np.array(D, D)
            right gauge with 2 legs,
            ordered left-right.
        Ar : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right-orthonormal
    """

    D = A.shape[0]
    d = A.shape[1]
    tol = max(tol, 1e-14)
    i = 1

    # Random guess for R0 if none specified
    if R0 is None:
        R0 = np.random.rand(D, D)

    # Normalize R0
    R0 = R0 / np.linalg.norm(R0)

    # Initialize loop
    R, Ar = rqPos(np.reshape(ncon((A, R0), ([-1, -2, 1], [1, -3])), (D, D * d)))
    R = R / np.linalg.norm(R)
    convergence = np.linalg.norm(R - R0)

    # Decompose A*R until R converges
    while convergence > tol:
        # calculate AR and decompose
        Rnew, Ar = rqPos(np.reshape(ncon((A, R), ([-1, -2, 1], [1, -3])), (D, D * d)))

        # normalize new R
        Rnew = Rnew / np.linalg.norm(Rnew)

        # calculate convergence criterium
        convergence = np.linalg.norm(Rnew - R)
        R = Rnew

        # check if iterations exceeds maxIter
        if i > maxIter:
            print("Warning, right decomposition has not converged ", convergence)
            break
        i += 1

    return R, Ar.reshape((D, d, D))


def qrPos(A):
    """
    Do a QR decomposition with positive diagonal elements for R.
        Parameters
        ----------
        A : np.array(M, N)
            Matrix to decompose.
        Returns
        -------
        Q : np.array(M, N)
            Orthogonal matrix.
        R : np.array(N, N)
            Upper triangular matrix,
            positive diagonal elements.
        Complexity
        ----------
        ~O(max(M, N) ** 3) algorithm.
    """

    M, N = A.shape

    # QR decomposition, scipy conventions: Q.shape = (M, M), R.shape = (M, N)
    Q, R = qr(A)

    # Throw out zeros under diagonal: Q.shape = (M, N), R.shape = (N, N)
    Q = Q[:, :N]
    R = R[:N, :]

    # extract signs and multiply with signs on diagonal
    diagSigns = np.diag(np.sign(np.diag(R)))
    Q = np.dot(Q, diagSigns)
    R = np.dot(diagSigns, R)

    return Q, R


def leftOrthonormalize(A, L0=None, tol=1e-14, maxIter=1e5):
    """
    Transform A to left-orthonormal gauge.
        Parameters
        ----------
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.
        L0 : np.array(D, D), optional
            Left gauge matrix,
            initial guess.
        tol : float, optional
            convergence criterium,
            norm(R - Rnew) < tol.
        maxIter : int
            maximum amount of iterations.
        Returns
        -------
        L : np.array(D, D)
            left gauge with 2 legs,
            ordered left-right.
        Al : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            left-orthonormal
    """

    D = A.shape[0]
    d = A.shape[1]
    tol = max(tol, 1e-14)
    i = 1

    # Random guess for L0 if none specified
    if L0 is None:
        L0 = np.random.rand(D, D)

    # Normalize L0
    L0 = L0 / np.linalg.norm(L0)

    # Initialize loop
    Al, L = qrPos(np.reshape(ncon((L0, A), ([-1, 1], [1, -2, -3])), (D * d, D)))
    L = L / np.linalg.norm(L)
    convergence = np.linalg.norm(L - L0)

    # Decompose L*A until L converges
    while convergence > tol:
        # calculate LA and decompose
        Al, Lnew = qrPos(np.reshape(ncon((L, A), ([-1, 1], [1, -2, -3])), (D * d, D)))

        # normalize new L
        Lnew = Lnew / np.linalg.norm(Lnew)

        # calculate convergence criterium
        convergence = np.linalg.norm(Lnew - L)
        L = Lnew

        # check if iterations exceeds maxIter
        if i > maxIter:
            print("Warning, left decomposition has not converged ", convergence)
            break
        i += 1

    return L, Al.reshape((D, d, D))


def mixedCanonical(A, L0=None, R0=None, tol=1e-14, maxIter=1e5):
    """
    Bring MPS tensor into mixed gauge, such that -Al-C- = -C-Ar- = Ac.
        Parameters
        ----------
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.
        Returns
        -------
        Al : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            left orthonormal.
        Ac : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauge.
        Ar : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right orthonormal.
        C : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right,
            diagonal.
        Complexity
        ----------
        O(D ** 3) algorithm.
    """

    D = A.shape[0]
    tol = max(tol, 1e-14)

    # Random guess for  L0 if none specified
    if L0 is None:
        L0 = np.random.rand(D, D)

    # Random guess for  R0 if none specified
    if R0 is None:
        R0 = np.random.rand(D, D)

    # Compute left and right orthonormal forms
    L, Al = leftOrthonormalize(A, L0, tol, maxIter)
    R, Ar = rightOrthonormalize(A, R0, tol, maxIter)

    # center matrix C is matrix multiplication of L and R
    C = L @ R

    # singular value decomposition to diagonalize C
    U, S, Vdag = svd(C)
    C = np.diag(S)

    # absorb corresponding unitaries in Al and Ar
    Al = ncon((np.conj(U).T, Al, U), ([-1, 1], [1, -2, 2], [2, -3]))
    Ar = ncon((Vdag, Ar, np.conj(Vdag).T), ([-1, 1], [1, -2, 2], [2, -3]))

    # normalize center matrix
    norm = np.trace(C @ np.conj(C).T)
    C /= np.sqrt(norm)

    # compute center MPS tensor
    Ac = ncon((Al, C), ([-1, -2, 1], [1, -3]))

    return Al, Ac, Ar, C

if __name__=="__main__":
    d = 2
    D = 8

    A = createMPS(D, d)
    A = normalizeMPS(A)

    AL, AR, AC, C = mixedCanonical(A)

    ALAL = ncon([AL, AL.conj()], ((1, 2, -1), (1, 2, -2)))
    I = np.eye(D)

    print(np.allclose(ALAL, I))
