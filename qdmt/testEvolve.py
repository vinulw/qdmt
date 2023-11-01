from evolve import *
from uMPSHelpers import *
from optimise import uniformToRhoN, uniformToRhoNDag
import numpy as np
from ncon import ncon
from scipy.stats import unitary_group

def testFirstOrderTrotterEvolve():
    d = 2
    D = 4
    U1 = unitary_group.rvs(d**2)
    U1ten = U1.reshape(d, d, d, d)
    U2 = unitary_group.rvs(d**2)
    U2ten = U2.reshape(d, d, d, d)
    I = np.eye(2, dtype=complex)

    A = createMPS(D, d)
    A = normalizeMPS(A)

    l, r = fixedPoints(A)

    # Check 4 site case
    N = 6 # Need extra sites on either side
    rho6 = uniformToRhoNDag(A, N, l=l, r=r).reshape(d**N, d**N)

    UUU = ncon([U1ten, U1ten, U1ten], [(-1, -2, -7, -8), (-3, -4, -9, -10), (-5, -6, -11, -12)]).reshape(d**N, d**N)
    IUUI = ncon([I, U2ten, U2ten, I], [(-1, -7), (-2, -3, -8, -9), (-4, -5, -10, -11), (-6, -12)]).reshape(d**N, d**N)

    evolvedRho6 = IUUI @ UUU @ rho6 @ UUU.conj().T @ IUUI.conj().T

    evolvedRho6 = evolvedRho6.reshape([d]*(2*N))
    evolvedRho6 = ncon([evolvedRho6,], ((1, -1, -2, -3, -4, 2, 1, -5, -6, -7, -8, 2),))

    evolvedTen = firstOrderTrotterEvolve(A, U1, U2, N=4, l=l, r=r)

    assert np.allclose(evolvedRho6, evolvedTen)
