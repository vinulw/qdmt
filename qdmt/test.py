from copy import copy
import numpy as np
from ncon import ncon

def isLeftCanonical(A):
    D, d, _ = A.shape

    lTM = ncon([A, A.conj()], ((1, 2, -1), (1, 2, -2)))

    return np.allclose(lTM, np.eye(D))
