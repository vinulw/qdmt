'''
This script prepares a ground state of the Transverse Field Ising model and
performs a quench across the quantum critical point.
'''
import sys
sys.path.insert(0, '../qdmt/')

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from tqdm import tqdm

from hamiltonian import TransverseIsing
from vumps import vumps
from evolve import firstOrderTrotterEvolve
from optimise import optimiseDensityGradDescent


###############################################################################
# Prepare some hyper parameters
###############################################################################

g1 = 1.5    # To prepare initial ground state
g2 = 0.2    # For quench Hamiltonian

dt = 0.1   # Time steps
maxTime = 1.5

D = 4   # Virtual dim
d = 2   # Physical dim
N = 4   # Number of sites for evolution

assert N%2 == 0, 'N has to be even'

###############################################################################
# Prepare a ground state tensor using VUMPS for a given g
###############################################################################
print('Preparing groundstate...')
h1 = TransverseIsing(1, g1, 2).reshape(d, d, d, d)
Al, Ac, Ar, C = vumps(h1, D, d, A0=None, tol=1e-8, tolFactor=1e-2,
                        verbose=False)
print('\tCompleted.')
A0 = Al     # Groundstate tensor

###############################################################################
# Perform the time evolution
###############################################################################
print('Initialising time evolution...')
h2 = TransverseIsing(1, g2, 2)
U = la.expm(-1j*h2*dt).reshape(d, d, d, d)


tRange = np.arange(0, maxTime, dt)
Ats = [A0]
errors = []
At = A0

print('Performing time evolution...')
for t in tqdm(tRange[1:]):
    rhotdt = firstOrderTrotterEvolve(At, U, U, N) # Evolve the state
    error, Atdt = optimiseDensityGradDescent(rhotdt, D, eps=1e-2, A0=At, tol=1e-5, maxIter=5e2)

    Ats.append(Atdt)
    errors.append(error)

    At = Atdt

plt.plot(tRange[1:], errors)
plt.show()
