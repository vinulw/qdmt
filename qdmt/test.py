from copy import copy
import numpy as np
from vumps import expValNMixed
from vumpt_tutorial import expVal2Mixed
from vumpt_tools import createMPS, normalizeMPS, mixedCanonical
from hamiltonian import Hamiltonian

if __name__=="__main__":
    d = 2
    D = 4
    A = createMPS(D, d)
    A = normalizeMPS(A)
    Al, Ac, Ar, C = mixedCanonical(A)

    H = Hamiltonian({'ZZ':-1, 'X':0.2}).to_matrix()
    H = H.reshape(2, 2, 2, 2)

    e2 = expVal2Mixed(H, Ac, Ar)
    print('E two site: ', e2)
    eN = expValNMixed(H, Ac, Ar)
    print('E N site: ', eN)
    print('Close? : ', np.allclose(e2, eN))

