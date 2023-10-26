from uMPSHelpers import createMPS, normalizeMPS
from optimise import *
import numpy as np
from ncon import ncon

def testGradient():
    d, D = 2, 4
    A = createMPS(D, d)
    A = normalizeMPS(A)

    rhoA = uniformToRho(A)
    grad = gradient(rhoA, A)

    assert np.allclose(np.linalg.norm(grad), 0)


def testUniformToRhoN():
    from uMPSHelpers import fixedPoints
    d, D = 2, 4
    A = createMPS(D, d)
    A = normalizeMPS(A)

    l, r = fixedPoints(A)
    rho2 = ncon([l, A, A, A.conj(), A.conj(), r],
                 ((2, 1), (1, -3, 4), (4, -4, 6), (2, -1, 7), (7, -2, 8), (6, 8)))

    rhoN = uniformToRhoN(A, 2)

    assert np.allclose(rho2, rhoN)

def testTraceDistance():

    d, D = 2, 4
    A = createMPS(D, d)
    A = normalizeMPS(A)

    rhoA = uniformToRhoN(A, 2)
    tDist = traceDistance(rhoA, rhoA)

    assert np.allclose(tDist, 0)

    rhoA = uniformToRhoN(A, 3)
    tDist = traceDistance(rhoA, rhoA)

    assert np.allclose(tDist, 0)

def testGradCenterTermsAB():
    d, D = 2, 4
    A = createMPS(D, d)
    A = normalizeMPS(A)

    l, r = fixedPoints(A)

    B = createMPS(D, d)
    B = normalizeMPS(B)

    rhoB = uniformToRhoN(A, 2)

    # calculate first contraction
    term1 = ncon((l, r, A, A, np.conj(A), rhoB), ([-1, 1], [5, 7], [1, 3, 2], [2, 4, 5], [-3, 6, 7], [3, 4, -2, 6]))

    # calculate second contraction
    term2 = ncon((l, r, A, A, np.conj(A), rhoB), ([6, 1], [5, -3], [1, 3, 2], [2, 4, 5], [6, 7, -1], [3, 4, 7, -2]))

    gradAB = term1 + term2

    assert np.allclose(gradAB, gradCenterTermsAB(rhoB, A, l=l, r=r))

