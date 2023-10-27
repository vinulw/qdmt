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

def testgradCenterTermsAB():
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


def testgradCenterTermsAA():
    d, D = 2, 4
    A = createMPS(D, d)
    A = normalizeMPS(A)

    l, r = fixedPoints(A)


    # Exact gradient for 2 site
    tensors = [l, A, A, A.conj(), r, l, A, A, A.conj(), A.conj(), r]
    edges = [
        (-1, 2), (2, 6, 10), (10, 12, 14), (-3, 11, 13), (14, 13),
        (4, 3), (3, -2, 8), (8, 11, 15), (4, 6, 9), (9, 12, 16), (15, 16)
    ]
    term1 = ncon(tensors, edges)

    tensors = [l, A, A, A.conj(), r, l, A, A, A.conj(), A.conj(), r]
    edges = [
        (1, 2), (2, 6, 10), (10, 12, 14), (1, 5, -1), (14, -3),
        (4, 3), (3, 5, 8), (8, -2, 15), (4, 6, 9), (9, 12, 16), (15, 16)
    ]
    term2 = ncon(tensors, edges)

    tensors = [l, A, A, A.conj(), A.conj(), r, l, A, A, A.conj(), r]
    edges = [
        (1, 2), (2, -2, 10), (10, 12, 14), (1, 5, 7), (7, 11, 13), (14, 13),
        (-1, 3), (3, 5, 8), (8, 11, 15), (-3, 12, 16), (15, 16)
    ]
    term3 = ncon(tensors, edges)

    tensors = [l, A, A, A.conj(), A.conj(), r, l, A, A, A.conj(), r]
    edges = [
        (1, 2), (2, 6, 10), (10, -2, 14), (1, 5, 7), (7, 11, 13), (14, 13),
        (4, 3), (3, 5, 8), (8, 11, 15), (4, 6, -1), (15, -3)
    ]
    term4 = ncon(tensors, edges)

    gradExact = term1 + term2 + term3 + term4

    gradAA = gradCenterTermsAA(A, 2, l, r)

    assert np.allclose(gradAA, gradExact)
