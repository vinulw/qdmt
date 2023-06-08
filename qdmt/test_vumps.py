import vumps as v
import numpy as np
from ncon import ncon

from vumpt_tutorial import expVal2Mixed, reducedHamMixed
from vumpt_tools import createMPS, normalizeMPS, mixedCanonical
from hamiltonian import Hamiltonian

from vumps import expValNMixed, rescaledHnMixed

def test_expvalNMixed():
    #TODO : Refactor so not comparing to tutorial code.
    # e.g. test known energies
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

    assert np.allclose(e2, eN)

def test_rescaledHMixed():
    d = 2
    D = 4
    A = createMPS(D, d)
    A = normalizeMPS(A)
    Al, Ac, Ar, C = mixedCanonical(A)

    H = Hamiltonian({'ZZ':-1, 'X':0.2}).to_matrix()
    H = H.reshape(2, 2, 2, 2)

    hN = rescaledHnMixed(H, Ac, Ar)
    h = reducedHamMixed(H, Ac, Ar)

    assert np.allclose(h, hN)

def test_Etilde():
    from vumps import Etilde
    from vumpt_tutorial import EtildeLeft, EtildeRight

    d = 2
    D = 4
    A = createMPS(D, d)
    A = normalizeMPS(A)
    Al, Ac, Ar, C = mixedCanonical(A)

    v = np.random.rand(D**2) # Take a random v to test

    # Calculate the left vs
    l = np.eye(D) # left fixed point of left transfer matrix: left orthonormal
    r = C @ np.conj(C).T # right fixed point of left transfer matrix

    vCorrectLeft = EtildeLeft(Al, l, r, v)

    ELmine = Etilde(Al, l.reshape(D**2), r.reshape(D**2))
    vMineLeft = ELmine.conj().T @ v

    # Calculate the right vs
    l = C.conj().T @ C
    r = np.eye(D)
    vCorrectRight = EtildeRight(Ar, l, r, v)
    ERmine = Etilde(Ar, l.reshape(-1), r.reshape(-1))
    vMineRight = ERmine @ v

    assert np.allclose(vMineLeft, vCorrectLeft)
    assert np.allclose(vMineRight, vCorrectRight)

def test_sumLeft():
    from vumps import sumLeft
    from vumpt_tutorial import LhMixed

    # Setup
    d = 2
    D = 4
    A = createMPS(D, d)
    A = normalizeMPS(A)
    Al, Ac, Ar, C = mixedCanonical(A)

    H = Hamiltonian({'ZZ':-1, 'X':0.2}).to_matrix()
    H = H.reshape(2, 2, 2, 2)

    htilde = rescaledHnMixed(H, Ac, Ar)
    tol = 1e-5

    # Compare Lh
    Lh_mine = sumLeft(Al, C, htilde, tol=tol)
    Lh_exact = LhMixed(htilde, Al, C, tol=tol)

    assert np.allclose(Lh_mine, Lh_exact.reshape(-1))
