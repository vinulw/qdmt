'''
Optimise the trace distance between state and density matrix.
'''
from ncon import ncon
import numpy as np
from uMPSHelpers import fixedPoints
from scipy.sparse.linalg import LinearOperator, gmres
from functools import partial

def uniformToRho(A, l=None, r=None):

    if l is None or r is None:
        l, r = fixedPoints(A)
    l, r = fixedPoints(A)
    rho = ncon([l, A, A, A.conj(), A.conj(), r],
                 ((2, 1), (1, -3, 4), (4, -4, 6), (2, -1, 7), (7, -2, 8), (6, 8)))
    return rho

def gradCenterTermsAB(rhoB, A, l=None, r=None):
    """
    Calculate the value of the center terms of $Tr(\rho_A \rho_B)$.
    """

    # calculate fixed points if not supplied
    if l is None or r is None:
        l, r = fixedPoints(A)

    # calculate first contraction
    term1 = ncon((l, r, A, A, np.conj(A), rhoB), ([-1, 1], [5, 7], [1, 3, 2], [2, 4, 5], [-3, 6, 7], [3, 4, -2, 6]))

    # calculate second contraction
    term2 = ncon((l, r, A, A, np.conj(A), rhoB), ([6, 1], [5, -3], [1, 3, 2], [2, 4, 5], [6, 7, -1], [3, 4, 7, -2]))

    return term1, term2

def gradCenterTermsAA(A, l=None, r=None):
    """
    Calculate the value of the center terms of $Tr(\rho_A \rho_A)$.
    """

    if l is None or r is None:
        l, r = fixedPoints(A)


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

    return term1, term2, term3, term4

def EtildeRight(A, l, r, v):
    """
    Implement the action of (1 - Etilde) on a right vector v.
    """

    D = A.shape[0]

    # reshape to matrix
    v = v.reshape(D, D)

    # transfermatrix contribution
    transfer = ncon((A, np.conj(A), v), ([-1, 2, 1], [-2, 2, 3], [1, 3]))

    # fixed point contribution
    fixed = np.trace(l @ v) * r

    # sum these with the contribution of the identity
    vNew = v - transfer + fixed

    return vNew.reshape((D ** 2))


def RhUniform(rhoB, A, l=None, r=None):
    """
    Find the partial contraction for Rh for $Tr(\rho_B \rho_A)$.
    """

    D = A.shape[0]

# if l, r not specified, find fixed points
    if l is None or r is None:
        l, r = fixedPoints(A)

# construct b, which is the matrix to the right of (1 - E)^P in the figure above
    b = ncon((r, A, A, np.conj(A), np.conj(A), rhoB), ([4, 5], [-1, 2, 1], [1, 3, 4], [-2, 8, 7], [7, 6, 5], [2, 3, 8, 6]))

# solve Ax = b for x
    A = LinearOperator((D ** 2, D ** 2), matvec=partial(EtildeRight, A, l, r))
    Rh = gmres(A, b.reshape(D ** 2))[0]

    return Rh.reshape((D, D))


def gradLeftTermsAB(rhoB, A, l=None, r=None):
    """
    Calculate the value of the left gradient terms for $Tr(\rho_A \rho_B)$.

    """

    # if l, r not specified, find fixed points
    if l is None or r is None:
        l, r = fixedPoints(A)

    # calculate partial contraction
    Rh = RhUniform(rhoB, A, l, r)

    # calculate full contraction
    leftTerms = ncon((Rh, A, l), ([1, -3], [2, -2, 1], [-1, 2]))

    return leftTerms

def gradLeftTermsAA(A, l=None, r=None):
    """
    Calculate the value of the left gradient terms for $Tr(\rho_A \rho_A)$.
    """

    # if l, r not specified, find fixed points
    if l is None or r is None:
        l, r = fixedPoints(A)

    rhoA = uniformToRho(A, l, r)

    return gradLeftTermsAB(rhoA, A, l, r)

def EtildeLeft(A, l, r, v):
    """
    Implement the action of (1 - Etilde) on a left vector v.
    """

    D = A.shape[0]

    # reshape to matrix
    v = v.reshape(D, D)

    # transfer matrix contribution
    transfer = ncon((v, A, np.conj(A)), ([3, 1], [1, 2, -2], [3, 2, -1]))

    # fixed point contribution
    fixed = np.trace(v @ r) * l

    # sum these with the contribution of the identity
    vNew = v - transfer + fixed

    return vNew.reshape((D ** 2))

def LhUniform(rhoB, A, l=None, r=None):
    """
    Find the partial contraction for Lh for $Tr(\rho_A \rho_B)$.
    """

    D = A.shape[0]

    # if l, r not specified, find fixed points
    if l is None or r is None:
        l, r = fixedPoints(A)

    # construct b, which is the matrix to the right of (1 - E)^P in the figure above
    b = ncon((l, A, A, np.conj(A), np.conj(A), rhoB), ([5, 1], [1, 3, 2], [2, 4, -2], [5, 6, 7], [7, 8, -1], [3, 4, 6, 8]))

    # solve Ax = b for x
    A = LinearOperator((D ** 2, D ** 2), matvec=partial(EtildeLeft, A, l, r))
    Lh = gmres(A, b.reshape(D ** 2))[0]

    return Lh.reshape((D, D))

def gradRightTermsAB(rhoB, A, l=None, r=None):
    """
    Calculate the value of the right terms for $Tr(\rho_A \rho_B)$.
    """

    # if l, r not specified, find fixed points
    if l is None or r is None:
        l, r = fixedPoints(A)

    # calculate partial contraction
    Lh = LhUniform(rhoB, A, l, r)

    # calculate full contraction
    rightTerms = ncon((Lh, A, r), ([-1, 1], [1, -2, 2], [2, -3]))

    return rightTerms

def gradRightTermsAA(A, l=None, r=None):
    """
    Calculate the value of the right terms for $Tr(\rho_A \rho_A)$.
    """
    if l is None or r is None:
        l, r = fixedPoints(A)

    rhoA = uniformToRho(A, l, r)

    return gradRightTermsAB(rhoA, A, l, r)

def gradient(rhoB, A, l=None, r=None):
    """
    Calculate the gradient of $Tr[(\rho_A-\rho_B)^2]$
    """

    # if l, r not specified, find fixed points
    if l is None or r is None:
        l, r = fixedPoints(A)

    # find terms
    centerTerm1, centerTerm2 = gradCenterTermsAB(rhoB, A, l, r)
    leftTerms = gradLeftTermsAB(rhoB, A, l, r)
    rightTerms = gradRightTermsAB(rhoB, A, l, r)

    centerTermAA1, centerTermAA2, centerTermAA3, centerTermAA4 = gradCenterTermsAA(A, l, r)
    leftTermsA = 2*gradLeftTermsAA(A, l, r)
    rightTermsA = 2*gradRightTermsAA(A, l, r)

    grad = centerTermAA1 + centerTermAA2 + centerTermAA3 + centerTermAA4 + leftTermsA + rightTermsA
    grad -= 2 * (centerTerm1 + centerTerm2 + leftTerms + rightTerms)

    return grad

if __name__=="__main__":
    from uMPSHelpers import createMPS, normalizeMPS
    d, D = 2, 4
    A = createMPS(D, d)
    A = normalizeMPS(A)

    rhoA = uniformToRho(A)

    grad = gradient(rhoA, A)

    print(grad)
    print(np.linalg.norm(grad))
