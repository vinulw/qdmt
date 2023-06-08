'''
Implementation of vumps from vumps tutorial

Source: VUMPs tutorial (https://github.com/leburgel/uniformMpsTutorial)

'''

import numpy as np
from vumpt_tools import createMPS, mixedCanonical, normalizeMPS
from vumpt_tools import rightOrthonormalize, leftOrthonormalize
from ncon import ncon
from functools import partial
from scipy.sparse.linalg import eigs, LinearOperator, gmres
from scipy.linalg import polar
from hamiltonian import Hamiltonian


def reducedHamMixed(h, Ac, Ar):
    """
    Regularize Hamiltonian such that its expectation value is 0.

        Parameters
        ----------
        h : np.array (d, d, d, d)
            Hamiltonian that needs to be reduced,
            ordered topLeft-topRight-bottomLeft-bottomRight.
        Ac : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauged.
        Ar : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right gauged.
        Returns
        -------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight.
    """

    d = Ac.shape[1]

    # calculate expectation value
    e = np.real(expVal2Mixed(h, Ac, Ar))

    # substract from hamiltonian
    hTilde = h - e * ncon((np.eye(d), np.eye(d)), ([-1, -3], [-2, -4]))

    return hTilde


def expVal2Mixed(O, Ac, Ar):
    """
    Calculate the expectation value of a 2-site operator in mixed gauge.
        Parameters
        ----------
        O : np.array(d, d, d, d)
            two-site operator,
            ordered topLeft-topRight-bottomLeft-bottomRight.
        Ac : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauged.
        Ar : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right gauged.
        Returns
        -------
        o : complex float
            expectation value of O.
    """

    # contract expectation value network
    o = ncon((Ac, Ar, np.conj(Ac), np.conj(Ar), O), ([1, 2, 3], [3, 4, 5], [1, 6, 7], [7, 8, 5], [2, 4, 6, 8]), order=[3, 2, 4, 1, 6, 5, 8, 7])

    return o

def LhMixed(hTilde, Al, C, tol=1e-5):
    """
    Calculate Lh, for a given MPS in mixed gauge.

        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalized.
        Al : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            left-orthonormal.
        C : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right.
        tol : float, optional
            tolerance for gmres

        Returns
        -------
        Lh : np.array(D, D)
            result of contraction,
            ordered bottom-top.

    """

    D = Al.shape[0]
    tol = max(tol, 1e-14)

    # construct fixed points for Al
    l = np.eye(D) # left fixed point of left transfer matrix: left orthonormal
    r = C @ np.conj(C).T # right fixed point of left transfer matrix

    # construct b
    b = ncon((Al, Al, np.conj(Al), np.conj(Al), hTilde), ([4, 2, 1], [1, 3, -2], [4, 5, 6], [6, 7, -1], [2, 3, 5, 7]))


    # solve Ax = b for x
    A = LinearOperator((D ** 2, D ** 2), matvec=partial(EtildeLeft, Al, l, r))
    Lh = gmres(A, b.reshape(D ** 2), tol=tol)[0]

    return Lh.reshape((D, D))

def EtildeLeft(A, l, r, v):
    """
    Implement the action of (1 - Etilde) on a left vector matrix v.

        Parameters
        ----------
        A : np.array (D, d, D)
            normalized MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalized.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalized.
        v : np.array(D**2)
            right matrix of size (D, D) on which
            (1 - Etilde) acts,
            given as a vector of size (D**2,)

        Returns
        -------
        vNew : np.array(D**2)
            result of action of (1 - Etilde)
            on a left matrix,
            given as a vector of size (D**2,)
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

def RhMixed(hTilde, Ar, C, tol=1e-5):
    """
    Calculate Rh, for a given MPS in mixed gauge.

        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalized.
        Ar : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right-orthonormal.
        C : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right.
        tol : float, optional
            tolerance for gmres

        Returns
        -------
        Rh : np.array(D, D)
            result of contraction,
            ordered top-bottom.
    """

    D = Ar.shape[0]
    tol = max(tol, 1e-14)

    # construct fixed points for Ar
    l = np.conj(C).T @ C # left fixed point of right transfer matrix
    r = np.eye(D) # right fixed point of right transfer matrix: right orthonormal

    # construct b
    b = ncon((Ar, Ar, np.conj(Ar), np.conj(Ar), hTilde), ([-1, 2, 1], [1, 3, 4], [-2, 7, 6], [6, 5, 4], [2, 3, 7, 5]))

    # solve Ax = b for x
    A = LinearOperator((D ** 2, D ** 2), matvec=partial(EtildeRight, Ar, l, r))
    Rh = gmres(A, b.reshape(D ** 2), tol=tol)[0]

    return Rh.reshape((D, D))


def EtildeRight(A, l, r, v):
    """
    Implement the action of (1 - Etilde) on a right vector v.

        Parameters
        ----------
        A : np.array (D, d, D)
            normalized MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalized.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalized.
        v : np.array(D**2)
            right matrix of size (D, D) on which
            (1 - Etilde) acts,
            given as a vector of size (D**2,)

        Returns
        -------
        vNew : np.array(D**2)
            result of action of (1 - Etilde)
            on a right matrix,
            given as a vector of size (D**2,)
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

def gradientNorm(hTilde, Al, Ac, Ar, C, Lh, Rh):
    """
    Calculate the norm of the gradient.

        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalized.
        Al : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            left orthonormal.
        Ar : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right orthonormal.
        Ac : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauge.
        C : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right.
        Lh : np.array(D, D)
            left environment,
            ordered bottom-top.
        Rh : np.array(D, D)
            right environment,
            ordered top-bottom.

        Returns
        -------
        norm : float
            norm of the gradient @Al, Ac, Ar, C
    """

    # calculate update on Ac and C using maps H_Ac and H_c
    AcUpdate = H_Ac(hTilde, Al, Ar, Lh, Rh, Ac)
    CUpdate = H_C(hTilde, Al, Ar, Lh, Rh, C)
    AlCupdate = ncon((Al, CUpdate), ([-1, -2, 1], [1, -3]))

    norm = np.linalg.norm(AcUpdate - AlCupdate)

    return norm

def H_Ac(hTilde, Al, Ar, Lh, Rh, v):
    """
    Action of the effective Hamiltonian for Ac (131) on a vector.
        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalized.
        Al : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            left-orthonormal.
        Ar : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right-orthonormal.
        Lh : np.array(D, D)
            left environment,
            ordered bottom-top.
        Rh : np.array(D, D)
            right environment,
            ordered top-bottom.
        v : np.array(D, d, D)
            Tensor of size (D, d, D)
        Returns
        -------
        H_AcV : np.array(D, d, D)
            Result of the action of H_Ac on the vector v,
            representing a tensor of size (D, d, D)
    """

    # first term
    term1 = ncon((Al, v, np.conj(Al), hTilde), ([4, 2, 1], [1, 3, -3], [4, 5, -1], [2, 3, 5, -2]))

    # second term
    term2 = ncon((v, Ar, np.conj(Ar), hTilde), ([-1, 2, 1], [1, 3, 4], [-3, 5, 4], [2, 3, -2, 5]))

    # third term
    term3 = ncon((Lh, v), ([-1, 1], [1, -2, -3]))

    # fourth term
    term4 = ncon((v, Rh), ([-1, -2, 1], [1, -3]))

    # sum
    H_AcV = term1 + term2 + term3 + term4

    return H_AcV


def H_C(hTilde, Al, Ar, Lh, Rh, v):
    """
    Action of the effective Hamiltonian for Ac (131) on a vector.
        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalized.
        Al : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            left-orthonormal.
        Ar : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right-orthonormal.
        Lh : np.array(D, D)
            left environment,
            ordered bottom-top.
        Rh : np.array(D, D)
            right environment,
            ordered top-bottom.
        v : np.array(D, D)
            Matrix of size (D, D)
        Returns
        -------
        H_CV : np.array(D, D)
            Result of the action of H_C on the matrix v.
    """

    # first term
    term1 = ncon((Al, v, Ar, np.conj(Al), np.conj(Ar), hTilde), ([5, 3, 1], [1, 2], [2, 4, 7], [5, 6, -1], [-2, 8, 7], [3, 4, 6, 8]))

    # second term
    term2 = Lh @ v

    # third term
    term3 = v @ Rh

    # sum
    H_CV = term1 + term2 + term3

    return H_CV

def calcNewCenter(hTilde, Al, Ac, Ar, C, Lh=None, Rh=None, tol=1e-5):
    """
    Find new guess for Ac and C as fixed points of the maps H_Ac and H_C.

        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalized.
        Al : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            left orthonormal.
        Ar : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right orthonormal.
        Ac : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauge.
        C : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right,
            diagonal.
        Lh : np.array(D, D)
            left environment,
            ordered bottom-top.
        Rh : np.array(D, D)
            right environment,
            ordered top-bottom.
        tol : float, optional
            current tolerance

        Returns
        -------
        AcTilde : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauge.
        CTilde : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right.
    """

    D = Al.shape[0]
    d = Al.shape[1]
    tol = max(tol, 1e-14)

    # calculate left en right environment if they are not given
    if Lh is None:
        Lh = LhMixed(hTilde, Al, C, tol)
    if Rh is None:
        Rh = RhMixed(hTilde, Ar, C, tol)

    # calculate new AcTilde

    # wrapper around H_Ac that takes and returns a vector
    handleAc = lambda v: (H_Ac(hTilde, Al, Ar, Lh, Rh, v.reshape(D, d, D))).reshape(-1)
    # cast to linear operator
    handleAcLO = LinearOperator((D ** 2 * d, D ** 2 * d), matvec=handleAc)
    # compute eigenvector
    _, AcTilde = eigs(handleAcLO, k=1, which="SR", v0=Ac.reshape(-1), tol=tol)


    # calculate new CTilde

    # wrapper around H_C that takes and returns a vector
    handleC = lambda v: (H_C(hTilde, Al, Ar, Lh, Rh, v.reshape(D, D))).reshape(-1)
    # cast to linear operator
    handleCLO = LinearOperator((D ** 2, D ** 2), matvec=handleC)
    # compute eigenvector
    _, CTilde = eigs(handleCLO, k=1, which="SR", v0=C.reshape(-1), tol=tol)

    # reshape to tensors of correct size
    AcTilde = AcTilde.reshape((D, d, D))
    CTilde = CTilde.reshape((D, D))

    return AcTilde, CTilde

def minAcC(AcTilde, CTilde, tol=1e-5):
    """
    Find Al and Ar corresponding to Ac and C, according to algorithm 5 in the lecture notes.

        Parameters
        ----------
        AcTilde : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            new guess for center gauge.
        CTilde : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right,
            new guess for center gauge

        Returns
        -------
        Al : np.array(D, d, D)
            MPS tensor zith 3 legs,
            ordered left-bottom-right,
            left orthonormal.
        Ar : np.array(D, d, D)
            MPS tensor zith 3 legs,
            ordered left-bottom-right,
            right orthonormal.
        Ac : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauge.
        C : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right,
            center gauge

    """

    D = AcTilde.shape[0]
    d = AcTilde.shape[1]
    tol = max(tol, 1e-14)

    # polar decomposition of Ac
    UlAc, _ = polar(AcTilde.reshape((D * d, D)))

    # polar decomposition of C
    UlC, _ = polar(CTilde)

    # construct Al
    Al = (UlAc @ np.conj(UlC).T).reshape(D, d, D)

    # find corresponding Ar, C, and Ac through right orthonormalizing Al
    C, Ar = rightOrthonormalize(Al, CTilde, tol=tol)
    nrm = np.trace(C @ np.conj(C).T)
    C = C / np.sqrt(nrm)
    Ac = ncon((Al, C), ([-1, -2, 1], [1, -3]))

    return Al, Ac, Ar, C


def vumps(h, D, A0=None, tol=1e-4, tolFactor=1e-1, verbose=True):
    """
    Find the ground state of a given Hamiltonian using VUMPS.

        Parameters
        ----------
        h : np.array (d, d, d, d)
            Hamiltonian to minimize,
            ordered topLeft-topRight-bottomLeft-bottomRight.
        D : int
            Bond dimension
        A0 : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            initial guess.
        tol : float
            Relative convergence criterium.

        Returns
        -------
        E : float
            expectation value @ minimum
        Al : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            left orthonormal.
        Ar : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right orthonormal.
        Ac : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauge.
        C : np.array (D, D)
            Center gauge with 2 legs,
            ordered left-right.
    """

    d = h.shape[0]

    # if no initial guess, random one
    if A0 is None:
        A0 = createMPS(D, d)

    # go to mixed gauge
    Al, Ac, Ar, C = mixedCanonical(A0)

    flag = True
    delta = 1e-5
    i = 0

    while flag:
        i += 1

        # regularize H
        hTilde = reducedHamMixed(h, Ac, Ar)

        # calculate environments
        Lh = LhMixed(hTilde, Al, C, tol=delta*tolFactor)
        Rh = RhMixed(hTilde, Ar, C, tol=delta*tolFactor)

        # calculate norm
        delta = gradientNorm(hTilde, Al, Ac, Ar, C, Lh, Rh)

        # check convergence
        if delta < tol:
            flag = False

        # calculate new center
        AcTilde, CTilde = calcNewCenter(hTilde, Al, Ac, Ar, C, Lh, Rh, tol=delta*tolFactor)

        # find Al, Ar from AcTilde, CTilde
        AlTilde, AcTilde, ArTilde, CTilde = minAcC(AcTilde, CTilde, tol=delta*tolFactor**2)

        # update tensors
        Al, Ac, Ar, C = AlTilde, AcTilde, ArTilde, CTilde

        # print current energy
        if verbose:
            E = np.real(expVal2Mixed(h, Ac, Ar))
            print('iteration:\t{:d}\tenergy:\t{:.12f}\tgradient norm:\t{:.4e}'.format(i, E, delta))

    return E, Al, Ac, Ar, C

if __name__=="__main__":
    d, D = 2, 8

    A = createMPS(D, d)
    A = normalizeMPS(A)
    H = Hamiltonian({'ZZ':-1, 'X':0.2}).to_matrix()

    H = H.reshape(2, 2, 2, 2)

    print('Energy optimisation using VUMPs')

    E, Al, Ac, Ar, C = vumps(H, D, A0=A, tol=1e-4, tolFactor=1e-2, verbose=True)

    print('Computer energy: ', E)


