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


def test_sumRight():
    from vumps import sumRight
    from vumpt_tutorial import RhMixed

    # Setup
    d = 2
    D = 4
    A = createMPS(D, d)
    A = normalizeMPS(A)
    Al, Ac, Ar, C = mixedCanonical(A)

    H = Hamiltonian({'ZZ':-1, 'X':0.2}).to_matrix()
    H = H.reshape(2, 2, 2, 2)

    print('Rescaled')
    htilde = rescaledHnMixed(H, Ac, Ar)
    tol = 1e-5

    # Compare Lh
    Rh_mine = sumRight(Ar, C, htilde, tol=tol)
    Rh_exact = RhMixed(htilde, Ar, C, tol=tol)

    assert np.allclose(Rh_mine, Rh_exact.reshape(-1))

def test_AcMap():
    from vumps import sumLeft, sumRight, construct_AcMap

    from vumpt_tutorial import H_Ac
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

    Lh = sumLeft(Al, C, htilde, tol=tol)
    Rh = sumRight(Ar, C, htilde, tol=tol)

    v = np.random.rand(d*D**2) + 1j*np.random.rand(d*D**2)

    # Compare H_Ac
    Lh = Lh.reshape(D, D)
    Rh = Rh.reshape(D, D)
    myH_Ac = construct_AcMap(Al, Ar, htilde, Lh, Rh)
    mine = myH_Ac @ v

    v = v.reshape(D, d, D)
    correct = H_Ac(htilde, Al, Ar, Lh, Rh, v)

    assert np.allclose(correct.reshape(-1), mine)

def test_CMap():
    from vumps import sumLeft, sumRight, construct_CMap

    from vumpt_tutorial import H_C
    # Setup
    d = 2
    D = 4
    A = createMPS(D, d)
    A = normalizeMPS(A)
    Al, Ac, Ar, C = mixedCanonical(A)

    H = Hamiltonian({'ZZ':-1, 'X':0.2}).to_matrix()
    H = H.reshape(2, 2, 2, 2)

    print('Rescaled')
    htilde = rescaledHnMixed(H, Ac, Ar)
    tol = 1e-5

    Lh = sumLeft(Al, C, htilde, tol=tol)
    Rh = sumRight(Ar, C, htilde, tol=tol)

    v = np.random.rand(D**2) + 1j*np.random.rand(D**2)

    # Compare H_Ac
    Lh = Lh.reshape(D, D)
    Rh = Rh.reshape(D, D)
    myH_C = construct_CMap(Al, Ar, htilde, Lh, Rh)
    mine = myH_C @ v

    v = v.reshape(D, D)
    correct = H_C(htilde, Al, Ar, Lh, Rh, v)

    assert np.allclose(mine, correct.reshape(-1))

def test_updateAcC():
    from vumps import sumLeft, sumRight, update_Ac, update_C
    from vumpt_tutorial import calcNewCenter

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

    Lh = sumLeft(Al, C, htilde, tol=tol).reshape(D, D)
    Rh = sumRight(Ar, C, htilde, tol=tol).reshape(D, D)

    # Compare updaed Ac
    correctAc, correctC = calcNewCenter(htilde, Al, Ac, Ar, C, Lh, Rh, tol=tol)

    myAc = update_Ac(htilde, Al, Ac, Ar, C, Lh, Rh, tol=tol)
    myC = update_C(htilde, Al, Ac, Ar, C, Lh, Rh, tol=tol)

    assert np.allclose(correctC, myC), 'C doesnt match'

    assert np.allclose(correctAc, myAc), 'Ac doesnt match'
