'''
Optimise the trace distance between state and density matrix.
'''
from ncon import ncon
import numpy as np
from uMPSHelpers import fixedPoints
from scipy.sparse.linalg import LinearOperator, gmres
from functools import partial
from copy import deepcopy

def uniformToRho(A, l=None, r=None):

    if l is None or r is None:
        l, r = fixedPoints(A)
    l, r = fixedPoints(A)
    rho = ncon([l, A, A, A.conj(), A.conj(), r],
                 ((2, 1), (1, -3, 4), (4, -4, 6), (2, -1, 7), (7, -2, 8), (6, 8)))
    return rho


def uniformToRhoN(A, N, l=None, r=None):
    '''
    Generate $\rho(A)_N$ which is the density matrix over $N$ sites.
    '''
    if l is None or r is None:
        l, r = fixedPoints(A)
    l, r = fixedPoints(A)

    tensors = [l, *[A]*N, *[A.conj()]*N, r]
    edges = [(2, 1)]
    i = 3
    edgesA = [(1, -N-1, i)]
    edgesAdag = [(2, -1, i+1)]
    i += 2
    for j in range(N-1):
        edgesA.append((i-2, -N-2-j, i))
        edgesAdag.append((i-1, -2-j, i+1))
        i += 2

    i = edgesA[-1][2]
    j = edgesAdag[-1][2]
    edges = edges + edgesA + edgesAdag + [(i, j)]

    return ncon(tensors, edges)


def traceDistance(A, B):
    assert A.shape == B.shape

    d = A.shape[0]
    N = len(A.shape) // 2
    A = A.reshape(d**N, d**N)
    B = B.reshape(d**N, d**N)

    dist = A - B
    return np.real(np.trace(dist @ dist.conj().T))

def gradCenterTermsAB(rhoB, A, N=None, l=None, r=None):

    # calculate fixed points if not supplied
    if l is None or r is None:
        l, r = fixedPoints(A)

    if N is None:
        N = len(rhoB.shape) // 2

    grad = np.zeros(A.shape, dtype=complex)

    tensors = [l, r, *[A]*N, *[A.conj()]*(N-1), rhoB]
    for i in range(N):
        print(i)
        contr = gradientContraction(N, i)
        print(contr)
        gradTerm = ncon(tensors, contr)
        print()
        grad += gradTerm

    return grad

def contractionExpA(N):
    lList = range(1, 4*N+1, 4)
    cList = range(3, 4*N+3, 4)
    rList = range(5, 4*N+5, 4)
    return [list(e) for e in zip(lList, cList, rList)]

def contractionExpADag(N):
    lList = range(2, 4*N+2, 4)
    cList = range(4, 4*N+4, 4)
    rList = range(6, 4*N+6, 4)
    return [list(e) for e in zip(lList, cList, rList)]

def contractionExpH(N):
    return [list(range(3, 4*N+3, 4)) + list(range(4, 4*N+4, 4))]

def expectationContraction(N):
    '''
    Generate the contraction list for expectation of an N site Hamiltonian
    '''
    Acontr = contractionExpA(N)
    ADagcontr = contractionExpADag(N)
    lcontr = [(ADagcontr[0][0], Acontr[0][0])]
    rcontr = [(Acontr[-1][-1], ADagcontr[-1][-1])]
    hcontr = contractionExpH(N)
    return lcontr, rcontr, Acontr, ADagcontr, hcontr

def gradientContraction(N, i):
    '''
    Generate contraction list for the gradient at site i of an N site Hamiltonian
    '''
    Acontr = contractionExpA(N)
    ADagcontr = contractionExpADag(N)
    lcontr = [[ADagcontr[0][0], Acontr[0][0]]]
    rcontr = [[Acontr[-1][-1], ADagcontr[-1][-1]]]
    hcontr = contractionExpH(N)

    # Left edge
    if i == 0:
        lcontr[0][0] = -1
        hcontr[0][N] = -2
        ADagcontr[1][0] = -3

        ADagcontr = ADagcontr[1:]
        return lcontr + rcontr + Acontr + ADagcontr + hcontr
    if i == N-1:
        ADagcontr[-2][2] = -1
        hcontr[0][-1] = -2
        rcontr[-1][1] = -3

        ADagcontr = ADagcontr[:-1]
        return lcontr + rcontr + Acontr + ADagcontr + hcontr

    ADagcontr[i-1][2] = -1
    hcontr[0][N+i] = -2
    ADagcontr[i+1][0] = -3

    ADagcontr = ADagcontr[:i] + ADagcontr[i+1:]
    return lcontr + rcontr + Acontr + ADagcontr + hcontr

def genAContr(l0, p0, r0, s, N):
    lList = range(l0, s*N+l0, s)
    pList = range(p0, s*N+p0, s)
    rList = range(r0, s*N+r0, s)

    return [list(e) for e in zip(lList, pList, rList)]


def contractionTrAA(N):
    outerA = genAContr(1, 5, 7, 6, N)
    outerAdag = genAContr(4, 6, 10, 6, N)
    innerA = genAContr(3, 6, 9, 6, N)
    innerAdag = genAContr(2, 5, 8, 6, N)

    return outerA, outerAdag, innerA, innerAdag

def gradCenterTermsAA(A, N, l=None, r=None):

    outerA, outerAdag, innerA, innerAdag = contractionTrAA(N)

    grad = np.zeros(A.shape, dtype=complex)

    tensors = [l, l, *[A]*(2*N), *[A.conj()]*(2*N-1), r, r]

    outerl = [[outerAdag[0][0], outerA[0][0]]]
    innerl = [[innerAdag[0][0], innerA[0][0]]]

    outerr = [[outerA[-1][-1], outerAdag[-1][-1]]]
    innerr = [[innerA[-1][-1], innerAdag[-1][-1]]]

    for i in range(N):
        # Take grad of outer
        outerAdag_ = deepcopy(outerAdag)
        outerl_ = deepcopy(outerl)
        outerr_ = deepcopy(outerr)
        innerA_ = deepcopy(innerA)

        if i > 0:
            outerAdag_[i-1][2] = -1
        if i < N-1:
            outerAdag_[i+1][0] = -3

        outerAdag_ = outerAdag_[:i] + outerAdag_[i+1:]
        if i == 0:
            outerl_[0][0] = -1
        if i == N-1:
            outerr_[0][1] = -3

        innerA_[i][1] = -2

        contr = outerl_ + innerl + outerA + innerA_ + outerAdag_ + innerAdag + outerr_ + innerr

        grad += ncon(tensors, contr)

        # Take grad of inner
        innerAdag_ = deepcopy(innerAdag)
        innerl_ = deepcopy(innerl)
        innerr_ = deepcopy(innerr)
        outerA_ = deepcopy(outerA)

        if i > 0:
            innerAdag_[i-1][2] = -1
        if i < N-1:
            innerAdag_[i+1][0] = -3

        innerAdag_ = innerAdag_[:i] + innerAdag_[i+1:]
        if i == 0:
            innerl_[0][0] = -1
        if i == N-1:
            innerr_[0][1] = -3

        outerA_[i][1] = -2

        contr = outerl + innerl_ + outerA_ + innerA + outerAdag + innerAdag_ + outerr + innerr_

        grad += ncon(tensors, contr)

    return grad


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
    Find the partial contraction for Rh for $Tr(\rho_B \rho_A)$ for an N site $\rho_B$.
    """
    N = len(rhoB.shape) // 2

    D = A.shape[0]

# if l, r not specified, find fixed points
    if l is None or r is None:
        l, r = fixedPoints(A)

    lcontr, rcontr, Acontr, ADagcontr, hcontr = expectationContraction(N)

    Acontr[0][0] = -1
    ADagcontr[0][0] = -2

    tensors = [*[A]*N, *[A.conj()]*N, r, rhoB]
    contr = [*Acontr, *ADagcontr, *rcontr, *hcontr]

    # construct b, which is the matrix to the right of (1 - E)^P in the figure above
    b = ncon(tensors, contr)

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

def gradLeftTermsAA(A, N, l=None, r=None):
    """
    Calculate the value of the left gradient terms for $Tr(\rho_A \rho_A)$.
    """

    # if l, r not specified, find fixed points
    if l is None or r is None:
        l, r = fixedPoints(A)

    rhoA = uniformToRhoN(A, N, l=l, r=r)

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
    Find the partial contraction for Lh for $Tr(\rho_B \rho_A)$ for an N site $\rho_B$.
    """
    N = len(rhoB.shape) // 2

    D = A.shape[0]

# if l, r not specified, find fixed points
    if l is None or r is None:
        l, r = fixedPoints(A)

    lcontr, rcontr, Acontr, ADagcontr, hcontr = expectationContraction(N)

    Acontr[-1][-1] = -2
    ADagcontr[-1][-1] = -1

    tensors = [*[A]*N, *[A.conj()]*N, l, rhoB]
    contr = [*Acontr, *ADagcontr, *lcontr, *hcontr]

    b = ncon(tensors, contr)

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

def gradRightTermsAA(A, N, l=None, r=None):
    """
    Calculate the value of the right terms for $Tr(\rho_A \rho_A)$.
    """
    if l is None or r is None:
        l, r = fixedPoints(A)

    rhoA = uniformToRhoN(A, N, l=l, r=r)

    return gradRightTermsAB(rhoA, A, l, r)

def gradient(rhoB, A, l=None, r=None):
    """
    Calculate the gradient of $Tr[(\rho_A-\rho_B)^2]$
    """
    N = len(rhoB.shape) // 2

    # if l, r not specified, find fixed points
    if l is None or r is None:
        l, r = fixedPoints(A)

    # find terms
    centerTerms = gradCenterTermsAB(rhoB, A, N, l, r)
    leftTerms = gradLeftTermsAB(rhoB, A, l, r)
    rightTerms = gradRightTermsAB(rhoB, A, l, r)

    centerTermsAA = gradCenterTermsAA(A, N, l, r)
    leftTermsA = 2*gradLeftTermsAA(A, N, l, r)
    rightTermsA = 2*gradRightTermsAA(A, N, l, r)

    grad = centerTermsAA + leftTermsA + rightTermsA
    grad -= 2 * (centerTerms + leftTerms + rightTerms)

    return grad

def optimiseDensityGradDescent(rhoB, D, eps=1e-1, A0=None, tol=1e-4, maxIter=1e4, verbose=True):
    """
    Find the tensor $A$ to optimise $Tr[(\rho_A - \rho_B)^2]$ using gradient descent.

    """

    d = rhoB.shape[0]
    N = len(rhoB.shape) // 2

    # if no initial value, choose random
    if A0 is None:
        A0 = createMPS(D, d)
        A0 = normalizeMPS(A0)

    # calculate gradient
    g = gradient(rhoB, A0)

    A = A0

    i = 0

    while not(np.linalg.norm(g) < tol):
        # do a step
        A = A - eps * g
        A = normalizeMPS(A)
        i += 1

        if verbose and not(i % 50):
            #E = np.real(expVal2Uniform(h, A))
            rhoA = uniformToRhoN(A, N)
            E = traceDistance(rhoB, rhoA)
            print('iteration:\t{:d}\tdist:\t{:.12f}\tgradient norm:\t{:.4e}'.format(i, E, np.linalg.norm(g)))

        # calculate new gradient
        g = gradient(rhoB, A)

        if i > maxIter:
            print('Warning: gradient descent did not converge!')
            break

    # calculate ground state energy
    # E = np.real(expVal2Uniform(h, A))
    rhoA = uniformToRhoN(A, N)
    E = traceDistance(rhoB, rhoA)

    return E, A

if __name__=="__main__":
    from uMPSHelpers import createMPS, normalizeMPS
    d, D = 2, 2
    A = createMPS(D, d)
    A = normalizeMPS(A)

    rhoA = uniformToRho(A)

    grad = gradient(rhoA, A)

    print(grad)
    print(np.linalg.norm(grad))

    A0 = createMPS(D, d)
    A0 = normalizeMPS(A0)

    print('Trying gradient descent...')
    E1, A1 = optimiseDensityGradDescent(rhoA, D, eps=1e-1, A0=A0, tol=1e-5, maxIter=1e4)
    print('Computed trace dist:', E1, '\n')
