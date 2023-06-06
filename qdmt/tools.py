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
    print(S[:3])

    normL, l = leftFixedPoint(TM)
    normR, r = rightFixedPoint(TM)

    l = l.reshape(D, D)
    r = r.reshape(D, D)

    # Force l to be hermitian
    l = 0.5*(l + l.conj().T)
    r = 0.5*(r + r.conj().T)

    print('Checking l is hermitian...')
    print(np.allclose(l, l.conj().T))

    norm = np.real(ncon([l, r], ((1, 2), (1, 2))))
    sign = np.sign(norm)
    l = sign * l
    norm = sign * norm

    l = l / np.sqrt(norm)
    r = r / np.sqrt(norm)

    print('Checking left eigenvector...')
    _, leftVal = is_eigenvector(TM, l.reshape(-1), left=True)

    print('Checking right eigenvector...')
    _, rightVal = is_eigenvector(TM, r.reshape(-1))
    print(leftVal)
    print(rightVal)
    lold = np.copy(l)
    l /= np.real(leftVal)
    r /= np.real(rightVal)
    print('Rescaling worked...')
    print(np.allclose(l, lold))

    _, leftVal = is_eigenvector(TM, l.reshape(-1), left=True)
    print(leftVal)

    # Decompose l = L^† L
    U, S, V = la.svd(l)
    print('Checking V = U^h')
    print(np.allclose(V, U.conj().T))

    L = np.diag(np.sqrt(S)) @ V
    Linv = la.inv(L)
    print('Checking l == L^† L')
    print(np.allclose(l, L.conj().T @ L ))

    Lprime = L.conj()
    LdagL = ncon([Lprime, Lprime.conj()], ((1, -1), (1, -2)))
    print(np.allclose(LdagL, l))

    Lprimeinv = la.inv(Lprime)
    LLinv = ncon([LdagL, Lprimeinv, Lprimeinv.conj()], ((1, 2), (1, -1), (2, -2)))
    I = np.eye(D)
    print(np.allclose(LLinv, I))

    LLA = ncon([Lprime, Lprime.conj(), A, A.conj()], ((1, 2), (1, 3), (2, 4, -1), (3, 4, -2)))
    λs = LLA.reshape(-1) / l.reshape(-1)
    print(np.allclose(λs, λs[0]))
    print(λs[0])

    print('Checking inverse...')
    I = np.eye(D)
    print(np.allclose(L @ Linv, I))

    Al = ncon([Lprime, A, Lprimeinv], ((-1, 1), (1, -2, 2), (2, -3)))
    # Al = ncon([L, A, Linv], ((-1, 1), (1, -2, 2), (2, -3)))

    print('Checking Al condition...')
    ALAL = ncon([Al, Al.conj()], ((1, 2, -1), (1, 2, -2)))
    vZeros = np.isclose(I.reshape(-1), 0)
    notvZeros = np.invert(vZeros)
    λs = ALAL.reshape(-1)[notvZeros] / I.reshape(-1)[notvZeros]
    print(np.allclose(ALAL, I))
    print('Al Multiple of I...')
    print(np.allclose(λs, λs[0]))
    print(λs[0])

    print('Checking Ar condition...')
    # Decompose r = R^† R
    U, S, V = la.svd(r)
    print('Checking V = U^h')
    print(np.allclose(V, U.conj().T))

    R = np.diag(np.sqrt(S)) @ V
    Rinv = la.inv(R)

    print('R†R = r')
    print(np.allclose(R.conj().T @ R, r))

    Rprime = R.conj().T
    Rprimeinv = la.inv(Rprime)
    RdagR = ncon([Rprime, Rprime.conj()], ((-1, 1), (-2, 1)))
    print(np.allclose(RdagR, r))

    # Ar = ncon([Rinv, A, R], ((-1, 1), (1, -2, 2), (2, -3)))
    Ar = ncon([Rprimeinv, A, Rprime], ((-1, 1), (1, -2, 2), (2, -3)))

    ARAR = ncon([Ar, Ar.conj()], ((-1, 1, 2), (-2, 1, 2)))
    vZeros = np.isclose(I.reshape(-1), 0)
    notvZeros = np.invert(vZeros)
    λs = ARAR.reshape(-1)[notvZeros] / I.reshape(-1)[notvZeros]
    print(np.allclose(ALAL, I))
    print('Ar Multiple of I...')
    print(np.allclose(λs, λs[0]))
    print(λs[0])

    C = Lprime @ Rprime

    # Diagonlize the mixed gauge
    U, S, V = la.svd(C)
    C = np.diag(S)

    Al = ncon([U.conj().T, Al, U], ((-1, 1), (1, -2, 2), (2, -3)))
    Ar = ncon([V, Ar, V.conj().T], ((-1, 1), (1, -2, 2), (2, -3)))

    print("Checking the gauge condition...")
    CAR = ncon([C, Ar], ((-1, 1), (1, -2, -3)))
    ALC = ncon([Al, C], ((-1, -2, 1), (1, -3)))

    print(np.allclose(CAR, ALC))

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





