import numpy as np
from numpy import linalg as LA
from ncon import ncon

m = 8
d = 2

""" initialize the MPS tensors """
C = np.random.rand(m)
C = C / LA.norm(C)
AL = (LA.svd(np.random.rand(m * d, m), full_matrices=False)[0]).reshape(m, d, m)
AR = (LA.svd(np.random.rand(m * d, m), full_matrices=False)[0]).reshape(m, d, m)
AC = ncon([AL, np.diag(C)], [[-1, -2, 1], [1, -3]])
HL = np.zeros([m, m])
HR = np.zeros([m, m])
