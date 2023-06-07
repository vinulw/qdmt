import numpy as np
from ncon import ncon
import numpy.linalg as la
from scipy.sparse.linalg import eigs


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
    '''
    Normalise A such that the transfer matrix of A has a leading eigenvalue of 1.
    '''
    D, _, _ = A.shape
    TM = ncon([A, A.conj()], ((-1, 1, -3), (-2, 1, -4))).reshape(D**2, D**2)

    _, S, _ = la.svd(TM)

    return A / np.sqrt(S[0])

def leftFixedPoint(M):
    M = M.T
    w, v = eigs(M, k=1)
    return w[0], v[:, 0]


def rightFixedPoint(M):
    w, v = eigs(M, k=1)
    return w[0], v[:, 0]


def mixedCanonical(A):
    '''
    Calculate mixed canonical form of translationally invariant MPS represented by A.

        Parameters
        ----------
        A: np.array(D, d, D)
            MPS tensor ordered (L, σ, R)
        Returns
        -------
        Al: np.array(D, d, D)
            MPS tensor ordered (L, σ, R)
        C: np.array(D, D)
            Diagonal centre gauge tensor ordered (L, R).
        Ar: np.array(D, d, D)
            MPS tensor ordered (L, σ, R)
    '''
    D = A.shape[0]
    TM = ncon([A, A.conj()], ((-1, 1, -3), (-2, 1, -4))).reshape(D**2, D**2)

    _, S, _ = la.svd(TM)

    _, l = leftFixedPoint(TM)
    _, r = rightFixedPoint(TM)

    l = l.reshape(D, D)
    r = r.reshape(D, D)

    ## Force l to be hermitian
    #l = 0.5*(l + l.conj().T)
    #r = 0.5*(r + r.conj().T)

    #norm = np.real(ncon([l, r], ((1, 2), (1, 2))))
    #sign = np.sign(norm)
    #l = sign * l
    #norm = sign * norm

    l = l / np.linalg.norm(l)
    r = r / np.linalg.norm(r)

    # Decompose l = L^† L
    U, S, V = la.svd(l)
    L = np.diag(np.sqrt(S)) @ V
    L = L / np.linalg.norm(L)
    Lprime = L.conj()
    Lprimeinv = la.inv(Lprime)

    Al = ncon([Lprime, A, Lprimeinv], ((-1, 1), (1, -2, 2), (2, -3)))
    # Al = ncon([L, A, Linv], ((-1, 1), (1, -2, 2), (2, -3)))

    # Decompose r = R^† R
    U, S, V = la.svd(r)
    R = np.diag(np.sqrt(S)) @ V
    R = R / np.linalg.norm(R)
    Rprime = R.conj().T
    Rprimeinv = la.inv(Rprime)

    # Ar = ncon([Rinv, A, R], ((-1, 1), (1, -2, 2), (2, -3)))
    Ar = ncon([Rprimeinv, A, Rprime], ((-1, 1), (1, -2, 2), (2, -3)))

    C = Lprime @ Rprime

    # Diagonlize the mixed gauge
    U, S, V = la.svd(C)
    C = np.diag(S)

    Al = ncon([U.conj().T, Al, U], ((-1, 1), (1, -2, 2), (2, -3)))
    Ar = ncon([V, Ar, V.conj().T], ((-1, 1), (1, -2, 2), (2, -3)))

    return Al, Ar, C


def is_eigenvector(M, v, left=False):
    if left:
        Mv = v @ M
    else:
        Mv = M @ v

    vZeros = np.isclose(v, 0)
    λs = Mv / v
    λs = λs[np.invert(vZeros)]

    return np.allclose(λs, λs[0]), λs[0]


if __name__=="__main__":

    D = 8
    d = 2
    A = createMPS(D, d)
    A = normalizeMPS(A)

    mixedCanonical(A)





