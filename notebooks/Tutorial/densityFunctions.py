import numpy as np
from ncon import ncon

def uniformToRho(A):
    D, d, _ = A.shape
    TM = ncon([A, A.conj()], ((-1, 1, -3), (-2, 1, -4))).reshape(D**2, D**2)

    vals, vecs = np.linalg.eig(TM)
    r = vecs[:, 0]

    vals, vecs = np.linalg.eig(TM.T)
    l = vecs[:, 0]

    rho = ncon([l.reshape(D, D), A, A, A.conj(), A.conj(), r.reshape(D, D)],
              ((1, 2), (1, -1, 4), (4, -2, 6), (2, -3, 7), (7, -4, 8), (6, 8)))
    return rho

def traceDistance(A, B):
    assert A.shape == B.shape
    assert len(A.shape) == 4

    d = A.shape[0]
    A = A.reshape(d**2, d**2)
    B = B.reshape(d**2, d**2)

    dist = A - B
    return np.real(np.trace(dist @ dist.conj().T))
