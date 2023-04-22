import numpy as np
from numpy import linalg as LA
from ncon import ncon

from doVUMPS import doVUMPS
from hamiltonian import Hamiltonian
from vumps import random_mixed_gauge, normalise_A

m = 4
d = 2

""" initialize the MPS tensors """
C = np.random.rand(m)
C = C / LA.norm(C)
AL = (LA.svd(np.random.rand(m * d, m), full_matrices=False)[0]).reshape(m, d, m)
AR = (LA.svd(np.random.rand(m * d, m), full_matrices=False)[0]).reshape(m, d, m)
AC = ncon([AL, np.diag(C)], [[-1, -2, 1], [1, -3]])
HL = np.zeros([m, m])
HR = np.zeros([m, m])

AL = normalise_A(AL)
AR = normalise_A(AR)



h0 = Hamiltonian({'ZZ':-1, 'X':0.2}).to_matrix().reshape(d, d, d, d)

AL, C, AR, HL, HR = doVUMPS(AL, C, AR, h0, HL, HR, num_iter=100, update_mode='svd')


