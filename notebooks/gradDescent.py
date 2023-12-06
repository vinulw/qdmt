import numpy as np
from ncon import ncon

def uniformToRho(A):
    TM = ncon([A, A.conj()], ((-1, 1, -3), (-2, 1, -4))).reshape(D**2, D**2)

    vals, vecs = np.linalg.eig(TM)
    r = vecs[:, 0]

    vals, vecs = np.linalg.eig(TM.T)
    l = vecs[:, 0]

    rho = ncon([l.reshape(2, 2), A, A, A.conj(), A.conj(), r.reshape(2, 2)],
              ((1, 2), (1, -1, 4), (4, -2, 6), (2, -3, 7), (7, -4, 8), (6, 8)))
    return rho

