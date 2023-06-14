import numpy as np
from numpy import linalg as la
from ncon import ncon
from copy import copy
import os

from numpy.linalg import qr
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from scipy.linalg import polar
from scipy.sparse.linalg import bicgstab, LinearOperator, gmres
from numpy.linalg import solve

import matplotlib.pyplot as plt
from tqdm import tqdm
from hamiltonian import Hamiltonian, TransverseIsing

from vumpt_tools import createMPS, normalizeMPS, mixedCanonical


def random_mixed_gauge(D, d, normalise=True):
    '''
    Generate a random mixed canonical uMPS state with bond dimension D and
    physical dimension d

    Returns
    -------
    A  :  Normalised state tensor
    AL :  Left orthonormal tensor representing A
    AR :  Right orthonormal tensor representing A
    C  :  Central tensor singular values, use np.diag(C) to get full matrix
    '''

    A = createMPS(D, d)
    if normalise:
        A = normalizeMPS(A)
    AL, _, AR, C = mixedCanonical(A)
    return A, AL, AR, C


def expValNMixed(O, Ac, Ar):
    '''
    Calculae the expectation value of an N-site operator in mixed gauge.
        Parameters
        ----------
        O : np.array(d, d, ..., d, d)
            two-site operator,
            ordered top legs followed by bottom legs.
        Ac : np.array(D, d, D)
            MPS tensor with 3 legs, ordered left-bottom-right, center gauged.
        Ar : np.array(D, d, D)
            MPS tensor with 3 legs, ordered left-bottom-right, right gauged.
        Returns
        -------
        o : complex float
            expectation value of O.
    '''
    n = len(O.shape) // 2
    tensors = [Ac] + [Ar] * (n-1)  # Top tensors
    tensors += [Ac.conj()] + [Ar.conj()] * (n-1) + [O]  # Bottom tensors and O

    # Top tensor contractions
    contrTop = zip(range(1, 2*n, 2), range(2, 2*n+1, 2), range(3, 2*n+2, 2))
    contrTop = [list(c) for c in contrTop]
    # Bottom tensor contractions
    final = 2*n+1
    contrBot = zip(range(final, final + 2*n, 2),
                   range(final+1, final + 2 + 2*n, 2),
                   range(final+2, final + 3 + 2*n, 2))
    contrBot = [list(c) for c in contrBot]
    contrBot[0][0] = contrTop[0][0]  # Connect left
    contrBot[-1][-1] = contrTop[-1][-1]  # Connect right
    # O contr
    contrO = [list(range(2, 4*n+1, 2))]

    contr = contrTop + contrBot + contrO

    # TODO: Set order to be more efficient
    return ncon(tensors, contr)

def rescaledHnMixed(h, Ac, Ar):
    '''
    Rescale the Hamiltonian so that the expectation is 0.

        Parameters
        ----------
        h : np.array (d, d, ..., d, d)
            Hamiltonian that needs to be reduced, ordered with the top legs
            followed by the bottom legs.
        Ac : np.array(D, d, D)
            MPS tensor with 3 legs, ordered left-bottom-right, center gauged.
        Ar : np.array(D, d, D)
            MPS tensor with 3 legs, ordered left-bottom-right, right gauged.
        Returns
        -------
        hTilde : np.array (d, d, ..., d, d)
            rescaled Hamiltonian, ordered top legs then bottom legs.
    '''
    d = Ac.shape[1]
    n = len(h.shape) // 2

    e = np.real(expValNMixed(h, Ac, Ar))

    # Generate the identity tensor
    Id = np.eye(d)
    tensors = [Id] * n
    contr = list(zip(range(-1, -n-1, -1), range(-n-1, -2*n-1, -1)))
    I = ncon(tensors, contr)

    hTilde = h - e * I
    return hTilde

def update_C(hTilde, Al, Ac, Ar, C, Lh, Rh, tol=1e-5):
    tol = max(tol, 1e-14)
    D = Al.shape[0]

    def CMapOp(C_mat):
        C_map =  construct_CMap(Al, Ar, hTilde, Lh, Rh)
        return (C_map @ C_mat.reshape(-1)).flatten()

    COp = LinearOperator((D**2, D**2), matvec=CMapOp)
    C_prime = eigs(COp, k=1, which='SR', v0=C.flatten(),
                   ncv=None, maxiter=None, tol=tol)[1]

    return C_prime.reshape(D, D)

def update_Ac(hTilde, Al, Ac, Ar, C, Lh, Rh, tol=1e-5):
    tol = max(tol, 1e-14)
    D, d, _ = Al.shape

    def AcMapOp(AC):
        AC_map = construct_AcMap(Al, Ar, hTilde, Lh, Rh)

        return AC_map @ AC.reshape(-1)

    AC_Op = LinearOperator((d * D**2, d * D**2), matvec=AcMapOp)
    AC_prime = (eigs(AC_Op, k=1, which='SR', v0=Ac.flatten(),
            ncv=None, maxiter=None, tol=tol)[1]).reshape(D, d, D)
    return AC_prime


def Etilde(A, l, r):
    D = A.shape[0]

    # Create fixed point
    fixed = ncon([l, r], ((-2,),(-1,)))

    # Create transfer matrix
    transfer = ncon([A, A.conj()], ((-1, 1, -3), (-2, 1, -4))).reshape(D**2, D**2)

    # Put together with identity
    Etilde = np.eye(D**2) - transfer + fixed

    return Etilde #TODO: Not sure why the conj is necessary


def sumLeft(AL, C, h, tol=1e-8):
    tol = max(tol, 1e-14)
    D, d, _ = AL.shape
    m = len(h.shape) // 2

    h0_edge = list(range(1, 2*m+1))
    curr_i = 2*m+1
    edges_A = [[curr_i, m+1, curr_i + 1]]
    edges_A_dag = [[curr_i, 1, curr_i + 2]]
    curr_i = curr_i + 1

    for i in range(1, m):
        edges_A.append([curr_i, m+1+i, curr_i+2])
        edges_A_dag.append([curr_i+1, i+1, curr_i+3])
        curr_i += 2

    edges_A[-1][-1] = -1
    edges_A_dag[-1][-1] = -2

    edges = [*edges_A, h0_edge, *edges_A_dag]
    tensors = [*[AL] * m, h, *[AL.conj()]*m]

    Hl = ncon(tensors, edges)
    # To be used in Ax = b solver
    b = Hl.reshape(D**2).conj()

    l = np.eye(D).reshape(-1)
    r = (C @ C.conj().T).reshape(-1)
    Et = Etilde(AL, l, r)

    A_ = Et.conj().T # So that Ax = b
    mvec = lambda v: A_ @ v
    A = LinearOperator((D**2, D**2), matvec=mvec)


    Lh = gmres(A, b, tol=tol)[0]

    return Lh

def sumRight(AR, C, h, tol=1e-8):
    D, d, _ = AR.shape
    m = len(h.shape) // 2

    h0_edge = list(range(1, 2*m+1))
    curr_i = 2*m+1
    curr_i += 1
    edges_A = [[curr_i, m+1, curr_i + 2]]
    edges_A_dag = [[curr_i+1, 1, curr_i + 3]]
    curr_i = curr_i + 2

    for i in range(1, m):
        edges_A.append([curr_i, m+1+i, curr_i+2])
        edges_A_dag.append([curr_i+1, i+1, curr_i+3])
        curr_i += 2

    edges_A_dag[-1][-1] = curr_i
    edges_A[0][0] = -1
    edges_A_dag[0][0] = -2

    edges = edges_A + [h0_edge] + edges_A_dag
    tensors = [*[AR]*m, h, *[AR.conj()]*m]
    Hr = ncon(tensors, edges)
    # To be used in Ax = b solver
    b = Hr.reshape(D**2)

    l = (C.conj().T @ C).reshape(-1)
    r = np.eye(D).reshape(-1)
    Et = Etilde(AR, l, r)

    A_ = Et # So that Ax = b
    mvec = lambda v: A_ @ v
    A = LinearOperator((D**2, D**2), matvec=mvec)

    Rh = gmres(A, b, tol=tol)[0]
    return Rh

def construct_CMap(Al, Ar, h, LH, RH):
    D = Al.shape[0]
    C_map = np.zeros((D**2, D**2), dtype=complex)
    n_sites = len(h.shape) // 2

    contr_h = list(range(1, 2*n_sites+1))
    start_i = contr_h[-1] + 1
    contr_h = contr_h[n_sites:] + contr_h[:n_sites]

    for nL in range(1, n_sites):
        nR = n_sites - nL
        i_ = start_i
        h_i = 1
        h_i_dag = n_sites + 1
        contr_Al = [[i_, h_i , i_+1]]
        contr_Al_dag = [[i_, h_i_dag, i_+2]]

        i_ = i_+1
        h_i += 1
        h_i_dag += 1

        for _ in range(nL - 1):
            contr_Al.append([i_, h_i, i_+2])
            contr_Al_dag.append([i_+1, h_i_dag, i_+3])

            i_ += 2
            h_i += 1
            h_i_dag += 1

        contr_Al[-1][-1] = -3
        contr_Al_dag[-1][-1] = -1

        contr_Ar = []
        contr_Ar_dag = []

        for _ in range(nR):
            contr_Ar.append([i_, h_i, i_+2])
            contr_Ar_dag.append([i_+1, h_i_dag, i_+3])

            i_ += 2
            h_i += 1
            h_i_dag += 1

        contr_Ar[0][0] = -4
        contr_Ar_dag[0][0] = -2
        contr_Ar_dag[-1][-1] = contr_Ar[-1][-1]

        contr = contr_Al + contr_Al_dag + [contr_h] + contr_Ar + contr_Ar_dag
        tensors = [Al] * nL + [Al.conj()] * nL + [h] + [Ar] * nR + [Ar.conj()] * nR

        C_map_ = ncon(tensors, contr).reshape(D**2, D**2)

        C_map += C_map_

    I = np.eye(D, dtype=complex)
    term2 = ncon([LH, I], ((-1, -3), (-2, -4))).reshape(D**2, D**2)
    term3 = ncon([I, RH], ((-1, -3), (-4, -2))).reshape(D**2, D**2)
    C_map += term2 + term3

    return C_map

def construct_AcMap(AL, AR, h, LH, RH):
    D, d, _ = AL.shape
    dim = d*D**2


    n_sites = len(h.shape) // 2

    h_contr = list(range(1, 2*n_sites+1))
    start_i = h_contr[-1] + 1
    h_contr = h_contr[n_sites:] + h_contr[:n_sites]

    I = np.eye(D, dtype=complex)
    Id = np.eye(d, dtype=complex)

    term_sites = []
    for site in range(n_sites):
        i = start_i
        nL = site
        nR = n_sites - 1 - site

        h_i = 1
        h_i_dag = n_sites + 1

        Al_contr = []
        Al_dag_contr = []
        Ar_contr = []
        Ar_dag_contr = []

        for _ in range(nL):
            Al_contr.append([i, h_i, i + 2])
            Al_dag_contr.append([i+1, h_i_dag, i + 3])

            h_i += 1
            h_i_dag += 1
            i += 2

        # Skip over the site
        h_i += 1
        h_i_dag += 1

        for _ in range(nR):
            Ar_contr.append([i, h_i, i + 2])
            Ar_dag_contr.append([i+1, h_i_dag, i + 3])

            h_i += 1
            h_i_dag += 1
            i += 2

        h_contr_ = copy(h_contr)
        h_contr_[site] = -2
        h_contr_[n_sites + site] = -5

        if len(Al_contr) > 0:
            Al_dag_contr[0][0] = Al_contr[0][0]
            Al_contr[-1][-1] = -4
            Al_dag_contr[-1][-1] = -1

        if len(Ar_contr) > 0:
            Ar_dag_contr[-1][-1] = Ar_contr[-1][-1]
            Ar_contr[0][0] = -6
            Ar_dag_contr[0][0] = -3

        I_contr = []
        if site == 0:
            I_contr.append([-1, -4])
        elif site == n_sites-1:
            I_contr.append([-3, -6])
        nI = len(I_contr)

        contr = Al_contr + Al_dag_contr + [h_contr_] + Ar_contr + Ar_dag_contr + I_contr
        tensors = [AL]*nL + [AL.conj()]*nL + [h] + [AR]*nR + [AR.conj()]*nR + [I]*nI
        term_sites.append(ncon(tensors, contr).reshape(dim, dim))

    term3 = ncon([LH, Id, I], ((-1, -4), (-2, -5), (-3, -6))).reshape(dim, dim)
    term4 = ncon([I, Id, RH], ((-1, -4), (-2, -5), (-3, -6))).reshape(dim, dim)
    term4 = term4.conj()

    AcMap = term3 + term4
    for term in term_sites:
        AcMap += term

    return AcMap


def minAcCPolar(AcTilde, CTilde, tol=1e-5):
    from vumpt_tools import rightOrthonormalize
    D, d, _ = AcTilde.shape
    tol = max(tol, 1e-14)

    Ul_Ac, _ = polar(AcTilde.reshape(D*d, D), side='left')
    Ul_C, _ = polar(CTilde, side='left')

    Al = (Ul_Ac @ (Ul_C.conj().T)).reshape(D, d, D)

    # find Ar and C through rightOrthonormalize
    # TODO dig into the rightOrthonormalize and rewrite
    C, Ar = rightOrthonormalize(Al, CTilde, tol=tol)
    norm = np.trace(C @ C.conj().T) # normalise state
    C = C / np.sqrt(norm)

    Ac = ncon([Al, C], ((-1, -2, 1), (1, -3)))

    return Al, Ac, Ar, C

def errorL(hTilde, Al, Ac, Ar, C, Lh, Rh):
    """
    Calculate ϵL to check for convergence.
    """
    D, d, _ = Al.shape
    AcTilde = construct_AcMap(Al, Ar, hTilde, Lh, Rh) @ Ac.reshape(-1)
    AcTilde = AcTilde.reshape(D, d, D)

    CTilde = construct_CMap(Al, Ar, hTilde, Lh, Rh) @ C.reshape(-1)
    CTilde = CTilde.reshape(D, D)

    AlCTilde = ncon((Al, CTilde), ((-1, -2, 1), (1, -3)))

    return np.linalg.norm(AcTilde - AlCTilde)

def tensorOperator(O, d=2):
    '''
    Reshape operator O to a tensor with local space of size d.
    '''
    m = np.emath.logn(d, O.shape[0])
    assert np.mod(m, 1) == 0, "d does not match h shape"
    m = int(m)
    OTen = O.reshape(*[d] * 2*m)

    return OTen


def vumps(h, D, d, A0=None, tol=1e-5, tolFactor=1e-1, maxiter=100, verbose=False, callback=None):
    '''
    Perform vumps to optimise local hamiltonian h.
    '''

    if A0 is None:
        A0 = createMPS(D, d)
        A0 = normalizeMPS(A0)

    Al, Ac, Ar, C = mixedCanonical(A0)

    delta = tol*1e-2
    count = 0

    while maxiter > count:
        count += 1

        # Rescale Hamiltonian
        hTilde = rescaledHnMixed(h, Ac, Ar)

        # Calculate the environments
        Lh = sumLeft(Al, C,  hTilde, tol=tolFactor*delta).reshape(D, D)
        Rh = sumRight(Ar, C, hTilde, tol=tolFactor*delta).reshape(D, D)

        # Update Ac
        AcTilde = update_Ac(hTilde, Al, Ac, Ar, C, Lh, Rh, tol=tolFactor*delta)
        # Update C
        CTilde = update_C(hTilde, Al, Ac, Ar, C, Lh, Rh, tol=tolFactor*delta)

        # Find update tensors
        Al, Ac, Ar, C = minAcCPolar(AcTilde, CTilde, tol=delta*tolFactor**2)

        # Calculate errorL
        delta = errorL(hTilde, Al, Ac, Ar, C, Lh, Rh)
        if verbose:
            E = np.real(expValNMixed(h, Ac, Ar))
            print(f'iteration: {count}')
            print(f'   energy: {E}')
            print(f'   errorL: {delta}')

        if callback is not None:
            callback(count, Al, Ac, Ar, C, hTilde, Lh, Rh)
        if delta < tol:
            break

    return Al, Ac, Ar, C


if __name__ == "__main__":
    d = 2
    D = 2

    A = createMPS(D, d)
    A = normalizeMPS(A)

    n = 16
    nQb = 4
    g_range = np.linspace(0.0, 1.6, n)
    Es = np.zeros(n)

    fname = f'gstate_ising2_D{D}_qb{nQb}.npy'
    skip = False
    if os.path.exists(fname):
        Es = np.load(f'gstate_ising2_D{D}_qb{nQb}.npy')
        print(f'File found: {fname}\nSkipping...')
        skip = True

    if not skip:
        for i, g in tqdm(enumerate(g_range), total=n):
            # H = Hamiltonian({'ZZ':-1, 'X':g}).to_matrix()
            H = TransverseIsing(-1, g, nQb)
            h = tensorOperator(H, d=d)

            Al, Ac, Ar, C = vumps(h, D, d, A0=A, tol=1e-8, tolFactor=1e-2, verbose=False)
            E = np.real(expValNMixed(h, Ac, Ar))

            Es[i] = E

        np.save(fname, Es)

    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 14})
    plt.title(f'Ground state optimisation, nQB:{nQb}, D:{D}')
    plt.ylabel(r'$<\psi|h|\psi>$')
    plt.xlabel('g')
    plt.plot(g_range, Es, label='VUMPS')
    # plt.legend()
    plt.show()

