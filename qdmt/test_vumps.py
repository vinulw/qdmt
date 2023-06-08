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

def test_EtildeLeft():
    from vumps import EtildeLeft as EtildeLeft_mine
    from vumpt_tutorial import EtildeLeft

    d = 2
    D = 4
    A = createMPS(D, d)
    A = normalizeMPS(A)
    Al, Ac, Ar, C = mixedCanonical(A)

    l = np.eye(D) # left fixed point of left transfer matrix: left orthonormal
    r = C @ np.conj(C).T # right fixed point of left transfer matrix

    v = np.random.rand(D**2) # Take a random v to test

    vCorrect = EtildeLeft(Al, l, r, v)

    Etildemine = EtildeLeft_mine(Al, l.reshape(D**2), r.reshape(D**2))
    vMine = v @ Etildemine

    assert np.allclose(vMine, vCorrect)

