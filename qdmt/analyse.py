import numpy as np
from scipy.linalg import eig
from ncon import ncon

from uMPSHelpers import fixedPoints
from optimise import contractionTrAA

def right_fixed_point(E):
    '''
    Calculate the right fixed point of a transfer matrix E

    E.shape = (N, N)
    '''
    evals, evecs = eig(E)
    sort = sorted(zip(evals, evecs), key=lambda x: np.linalg.norm(x[0]),
                   reverse=True)
    # Look into `scipy.sparse.linalg.eigs, may be faster`
    mu, r = sort[0]
    return mu, r


def exact_overlap(A, B):
    '''
    Calculate the overlap classically i.e. abs of max eigenvalue of transfer
    matrix
    '''
    Dla, d, Dra = A.shape
    Dlb, _, Drb  = B.shape
    print('\tA shape: ', A.shape)
    print('\tB shape: ', B.shape)
    E = ncon([A, B.conj()], ((-1, 1, -3), (-2, 1, -4))).reshape(Dla*Dlb, Dra*Drb)

    mu, _ = right_fixed_point(E)

    return np.abs(mu)

def exact_overlap2(A, B):
    '''
    Calculate the overlap of a 2 site translationally invariant MPS.
    Classically i.e. abs of max eigenvalue of transfer
    matrix.
    '''
    Da, d, _ = A[0].shape
    Db, _, _  = B[0].shape
    E = ncon([A[0], A[1],
              B[0].conj(), B[1].conj()],
             ((-1, 1, 2), (2, 3, -3),
             (-2, 1, 4), (4, 3, -4))).reshape(Da*Db, Da*Db)

    mu, _ = right_fixed_point(E)

    return np.abs(mu)

def TrAB(A, B, N, fixedA=None, fixedB=None):
    if fixedA is None:
        fixedA = fixedPoints(A)
    if fixedB is None:
        fixedB = fixedPoints(B)
    lA, rA = fixedA
    lB, rB = fixedB

    BContr, BDagContr, AContr, ADagContr = contractionTrAA(N)

    lBContr = [[BDagContr[0][0], BContr[0][0]]]
    rBContr = [[BContr[-1][2], BDagContr[-1][2]]]

    lAContr = [[ADagContr[0][0], AContr[0][0]]]
    rAContr = [[AContr[-1][2], ADagContr[-1][2]]]

    contr = lBContr + BContr + BDagContr + rBContr +\
            lAContr + AContr + ADagContr + rAContr
    tensors = [lB] + [B]*N + [B.conj()]*N + [rB] +\
            [lA] + [A]*N + [A.conj()]*N + [rA]

    return ncon(tensors, contr)
