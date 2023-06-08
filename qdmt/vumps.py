import numpy as np
from numpy import linalg as la
from ncon import ncon
from copy import copy

from numpy.linalg import qr
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from scipy.linalg import polar
from scipy.sparse.linalg import bicgstab, LinearOperator, gmres
from numpy.linalg import solve

import matplotlib.pyplot as plt
from tqdm import tqdm
from hamiltonian import Hamiltonian

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

def normalise_A(A):
    '''
    Normalise A such that the transfer matrix of A has a leading eigenvalue of 1.
    '''
    D, _, _ = A.shape
    TM = ncon([A, A.conj()], ((-1, 1, -3), (-2, 1, -4))).reshape(D**2, D**2)

    _, S, _ = la.svd(TM)

    return A / np.sqrt(S[0])

def evaluateEnergy(AL, AR, C, h, debug=False):
    '''
    Evaluate the expectation for energy <h> for a uMPS state in mixed canonical
    form. Assume that h is a two site operator for now.

    TODO: Extend this to an n-site operator
    '''
    n_sites = h.shape[0]
    n_sites = int(np.log2(n_sites))
    h = h.reshape(*[2]*(n_sites*2))


    ACl = ncon([AL, C], ((-1, -2, 1), (1, -3)))
    ACr = ncon([C, AR], ((-1, 1), (1, -2, -3)))

    curr_contr_top = n_sites*2+2
    curr_contr_bot = n_sites*3+2
    curr_contr = n_sites*2+1
    curr_h_top = n_sites + 1
    curr_h_bot = 1
    contrAl = [(curr_contr, curr_h_top, curr_contr_top), ]
    contrAl_dag = [(curr_contr, curr_h_bot, curr_contr_bot), ]
    contr_h = tuple(range(1, n_sites*2+1))
    curr_h_top += 1
    curr_h_bot += 1
    for _ in range(n_sites - 1):
        contrAl.append((curr_contr_top, curr_h_top, curr_contr_top + 1))
        curr_h_top += 1
        curr_contr_top += 1

        contrAl_dag.append((curr_contr_bot, curr_h_bot, curr_contr_bot + 1))
        curr_contr_bot += 1
        curr_h_bot += 1

    # Connect the last leg
    finalAl = list(contrAl[-1])
    finalAl[2] = contrAl_dag[-1][2]
    contrAl[-1] = tuple(finalAl)

    Als = [AL] * (n_sites - 1)
    Als.append(ACl)

    Ars = [ACr] + [AR] * (n_sites - 1)

    Als_dag = [AL.conj()] * (n_sites - 1)
    Als_dag.append(ACl.conj())

    Ars_dag = [ACr.conj()] + [AR.conj()] * (n_sites - 1)

    energyL_new = ncon([*Als, h, *Als_dag], (*contrAl, contr_h, *contrAl_dag))

    energyR_new = ncon([*Ars, h, *Ars_dag], (*contrAl, contr_h, *contrAl_dag))

    if n_sites == 2 and debug:
        energyL = ncon([AL, AL.conj(), h, ACl, ACl.conj()],
                        ((1, 2, 3), (1, 4, 5), (4, 7, 2, 6), (3, 6, 8), (5, 7, 8)))


        energyR = ncon([ACr, ACr.conj(), h, AR, AR.conj()],
                        ((1, 2, 3), (1, 4, 5), (4, 7, 2, 6), (3, 6, 8), (5, 7, 8)))


        if debug:
            print('Checking 2 site energy...')
            print('   Left energy close....')
            print('   ', np.allclose(energyL_new, energyL))

            print('   Right energy close...')
            print('   ', np.allclose(energyR_new, energyR))

    energy = 0.5*(energyL_new + energyR_new) / n_sites # Density per site

    return energy, energyL_new, energyR_new

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
    tensors = [Ac] + [Ar] * (n-1) # Top tensors
    tensors += [Ac.conj()] + [Ar.conj()] * (n-1) + [O] # Bottom tensors and O

    # Top tensor contractions
    contrTop = zip(range(1, 2*n, 2), range(2, 2*n+1, 2), range(3, 2*n+2, 2))
    contrTop = [list(c) for c in contrTop]
    # Bottom tensor contractions
    final = 2*n+1
    contrBot = zip(range(final, final + 2*n, 2),
                   range(final+1, final + 2 + 2*n, 2),
                   range(final+2, final + 3 + 2*n, 2))
    contrBot = [list(c) for c in contrBot]
    contrBot[0][0] = contrTop[0][0] # Connect left
    contrBot[-1][-1] = contrTop[-1][-1] # Connect right
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

def gs_vumps(h, D, d, tol=1e-5, maxiter=100, strategy='polar', A0=None):
    '''
    Perform vumps to optimise local hamiltonian h.
    '''

    if A0 is None:
        A0 = createMPS(D, d)
        A0 = normalizeMPS(A0)

    AL, _, AR, C = mixedCanonical(A0)
    AC = ncon([AL, C], [[-1, -2, 1], [1, -3]])

    ev_tol = 1e-12 # Should be an optional
    h0 = h.copy()

    # Reshape the h (d**m, d**m ) -> (d, d, d, ..., d)
    m = np.emath.logn(d, h0.shape[0])
    assert np.mod(m, 1) == 0, "d does not match h shape"
    m = int(m)
    h0_ten = h0.reshape(*[d] * 2*m)

    print('h0_ten shape: ', h0_ten.shape)

    δ = 1
    count = 0
    energies = []
    error_δs = []
    error_ϵLs = []
    error_ϵRs = []

    if strategy == 'svd':
        minAcC = minAcC_svd
    else:
        minAcC = minAcC_polar

    while δ > tol and maxiter > count:
        e, _, _ = evaluateEnergy(AL, AR, C, h0)
        energies.append(e)
        print(f'Current energy : {e}')

        # Calculate the energy shifts (Left)
        h0_edge = list(range(1, 2*m+1))
        curr_i = 2*m+1
        edges_A = [[curr_i, m+1, curr_i + 1]]
        edges_A_dag = [[curr_i, 1, curr_i + 2]]
        curr_i = curr_i + 1

        for i in range(1, m):
            edges_A.append([curr_i, m+1+i, curr_i+2])
            edges_A_dag.append([curr_i+1, 1+i, curr_i+3])
            curr_i += 2

        edges = [*edges_A, h0_edge, *edges_A_dag, [curr_i, curr_i+2], [curr_i+1, curr_i+2]]
        tensors = [*[AL] * m, h0_ten, *[AL.conj()]*m, C, C.conj()]
        eL = ncon(tensors, edges)

        # Calculate the energy shift right
        curr_i = 2*m+1
        edges = [[curr_i, curr_i+1], [curr_i, curr_i+2]]
        curr_i += 1
        edges_A = [[curr_i, m+1, curr_i + 2]]
        edges_A_dag = [[curr_i+1, 1, curr_i + 3]]
        curr_i = curr_i + 2

        for i in range(1, m):
            edges_A.append([curr_i, m+1+i, curr_i+2])
            edges_A_dag.append([curr_i+1, i+1, curr_i+3])
            curr_i += 2

        edges_A_dag[-1][-1] = curr_i

        edges = edges + edges_A + [h0_edge] + edges_A_dag
        tensors = [C, C.conj(), *[AR]*m, h0_ten, *[AR.conj()]*m]
        eR = ncon(tensors, edges)

        h_shifted = (h - e*np.eye(d**m)).reshape(*[d]*2*m)
        h_shiftedL = (h - eL*np.eye(d**m)).reshape(*[d]*2*m)
        h_shiftedR = (h - eR*np.eye(d**m)).reshape(*[d]*2*m)

        LH = sumLeft(AL, h_shiftedL)
        RH = sumRight(AR, h_shiftedR)

        # Make them symmetric (should not be necessary)
        LH = 0.5*(LH + LH.T)
        RH = 0.5*(RH + RH.T)

        # Create identity maps
        I = np.eye(D)
        Id = np.eye(d)

        def CMapOp(C_mat):
            C_map =  construct_CMap(AL, AR, h0_ten, LH, RH, D)
            return (C_map @ C_mat.reshape(-1)).flatten()

        COp = LinearOperator((D**2, D**2), matvec=CMapOp, dtype=np.float64)
        C_prime = eigsh(COp, k=1, which='SA', v0=C.flatten(),
                       ncv=None, maxiter=None, tol=ev_tol)[1]


        # Convert to diagonal gauge for stability
        C_prime = C_prime.reshape(D, D)
        ut, C_prime, vt = la.svd(C_prime)
        C_prime = np.diag(C_prime)
        AL = ncon([ut.conj().T, AL, ut], [[-1, 1], [1, -2, 2], [2, -3]])
        AR = ncon([vt, AR, vt.conj().T], [[-1, 1], [1, -2, 2], [2, -3]])

        LH = ut.conj().T @ LH @ ut
        RH = vt @ RH @ vt.conj().T

        # Construct Ac′
        dim = d*D**2

        def AcMapOp(AC):
            AC_map = construct_AcMap(AL, AR, h0_ten, d, D, LH, RH)

            return AC_map @ AC.reshape(-1)

        print('Building AcMap')
        AC_Op = LinearOperator((d * D**2, d * D**2), matvec=AcMapOp, dtype=np.float64)
        AC_prime = (eigsh(AC_Op, k=1, which='SA', v0=AC.flatten(),
                ncv=None, maxiter=None, tol=ev_tol)[1]).reshape(D, d, D)

        #AL, AR, ϵL, ϵR = minAcC(AC_prime, C_prime, errors=True)

        if strategy == 'polar':
          AL = (polar(AC_prime.reshape(D * d, D))[0]).reshape(D, d, D)
          AR = (polar(AC_prime.reshape(D, d * D), side='left')[0]
                ).reshape(D, d, D)
        elif strategy == 'svd':
          ut, _, vt = la.svd(AC.reshape(D * d, D) @ C_prime, full_matrices=False)
          AL = (ut @ vt).reshape(D, d, D)
          ut, _, vt = la.svd(C_prime @ AC.reshape(m, d * m), full_matrices=False)
          AR = (ut @ vt).reshape(D, d, D)

        ALC = ncon([AL, C_prime], ((-1, -2, 1), (1, -3)))
        ϵL = np.linalg.norm(ALC - AC)
        CAR = ncon([C_prime, AR], ((-1, 1), (1, -2,  -3)))
        ϵR = np.linalg.norm(CAR - AC)

        C = C_prime.copy()
        AC = AC_prime.copy()


        # Check convergence
        # AC_map = AC_map.reshape(D, d, D, D, d, D)
        # C_map = C_map.reshape(D, D, D, D)
        # H_AC = ncon([AC_map, AC], ((1, 2, 3, -1, -2, -3), (1, 2, 3)))
        # H_C = ncon([C_map, C], ((1, 2, -1, -2), (1, 2)))
        # AL_HC = ncon([AL, H_C], ((-1, -2, 1), (1, -3)))

        δ = 1 # np.linalg.norm(H_AC - AL_HC)
        count += 1 # iteratre counter for maxiter

        error_δs.append(δ)
        error_ϵLs.append(ϵL)
        error_ϵRs.append(ϵR)

        print(f'Energy after opt: {e}')
        print('Errors: ')
        print(f'   δ: {δ}')
        print(f'   ϵL: {ϵL}')
        print(f'   ϵR: {ϵR}')

        # Normalise tensors before energy calculations
        AL = normalise_A(AL)
        AR = normalise_A(AR)
        AC = normalise_A(AC)
        e, _, _ = evaluateEnergy(AL, AR, C, h) # Calculate the final energy
        energies.append(e)

    plt.figure()
    plt.plot(error_δs)
    plt.title('Error δ')

    plt.figure()
    plt.plot(error_ϵLs, label='ϵL')
    plt.plot(error_ϵRs, label='ϵR')
    plt.legend()
    plt.title('Error ϵ{L/R}')

    e = evaluateEnergy(AL, AR, C, h) # Calculate the final energy
    return AL, AR, C, energies

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


def minAcC_svd(Ac, C, errors=False):
    print('Using svd strategy...')
    AcC = ncon([Ac, C.conj().T], ((-1, -2, 1), (1, -3))).reshape(d*D, D)
    Ul, _, Vl = la.svd(AcC, full_matrices=False)
    AL =  (Ul @ Vl).reshape(D, d, D)

    CAc = ncon([C.conj().T, Ac], ((-1, 1), (1, -2, -3))).reshape(D, d*D)
    Ur, _, Vr = la.svd(CAc, full_matrices=False)
    AR = (Ur @ Vr).reshape(D, d, D)

    print('Verigying Al canonical...')
    ALAL = ncon([AL, AL.conj()], ((1, 2, -1), (1, 2, -2)))
    print(np.allclose(ALAL, np.eye(D)))

    print('Verigying Ar canonical...')
    ARAR = ncon([AR, AR.conj()], ((-1, 1, 2), (-2, 1, 2)))
    print(np.allclose(ARAR, np.eye(D)))

    AlC = ncon([AL, C], ((-1, -2, 1), (1, -3)))
    ϵL = np.linalg.norm(AlC - Ac) # Error in Al, should converge near optima

    CAr = ncon([C, AR], ((-1, 1), (1, -2, -3)))
    ϵR = np.linalg.norm(CAr - Ac) # Error in Ar should converge near optima

    if errors:
        return AL, AR, ϵL, ϵR
    return AL, AR


def minAcC_polar(Ac, C, errors=False):
    Ul_Ac, _ = polar(Ac.reshape(D*d, D), side='left')
    Ul_C, _ = polar(C, side='left')

    Al = Ul_Ac @ (Ul_C.conj().T)
    Al = Al.reshape(D, d, D)

    Ur_Ac, _ = polar(Ac.reshape(D, D*d), side='right')
    Ur_C, _ = polar(C, side='right')

    Ar = (Ur_C.conj().T) @ Ur_Ac
    Ar = Ar.reshape(D, d, D)

    print('Verigying Al canonical...')
    AlAl = ncon([Al, Al.conj()], ((1, 2, -1), (1, 2, -2)))
    print(np.allclose(AlAl, np.eye(D)))

    print('Verigying Ar canonical...')
    ArAr = ncon([Ar, Ar.conj()], ((-1, 1, 2), (-2, 1, 2)))
    print(np.allclose(ArAr, np.eye(D)))

    AlC = ncon([Al, C], ((-1, -2, 1), (1, -3)))
    ϵL = np.linalg.norm(AlC - Ac) # Error in Al, should converge near optima

    CAr = ncon([C, Ar], ((-1, 1), (1, -2, -3)))
    ϵR = np.linalg.norm(CAr - Ac) # Error in Ar should converge near optima

    # Don't think I need to do all of this
    # # Diagonlise C and gauge transform Al and Ar
    # print('Diagonalising C')
    # U, C, V = la.svd(C)
    # C = np.diag(C)
    # Al = ncon([U.conj().T, Al, U], ((-1, 1), (1, -2, 2), (2, -3)))
    # # Ar = ncon([V, Ar, V.conj().T], ((-1, 1), (1, -2, 2), (2, -3)))

    # print('Verigying Al canonical...')
    # AlAl = ncon([Al, Al.conj()], ((1, 2, -1), (1, 2, -2)))
    # print(np.allclose(AlAl, np.eye(D)))

    # print('Verigying Ar canonical...')
    # ArAr = ncon([Ar, Ar.conj()], ((-1, 1, 2), (-2, 1, 2)))
    # print(np.allclose(ArAr, np.eye(D)))

    # print('Verifying Ac condition...')
    # Ac = ncon([Al, C], ((-1, -2, 1), (1, -3)))
    # Acr = ncon([C, Ar], ((-1, 1), (1, -2, -3)))
    # print(np.linalg.norm(Ac - Acr ))

    if errors:
        return Al, Ar, ϵL, ϵR
    return Al, Ar

def largest_evec_left(E, l0 = None, eval=False):
    '''
    Find leading eigenvector v of E such that vE = λv
    '''
    Eh = E.conj().T
    if l0 is not None:
        l0 = l0.conj().transpose()

    w, v = eigs(Eh, k=1, which='LM')

    e = v[:, 0]
    e = e.conj().transpose()

    if eval:
        return e, w[0]

    return e

def largest_evec_right(E, r0 = None, eval=False):
    '''
    Find leading eigenvector v of E such that Ev = λv
    '''
    w, v = eigs(E, k=1, which='LM', v0=r0)

    e = v[:, 0]

    if eval:
        return e, w[0]

    return e

if __name__=="__main__":
    d = 2
    D = 4
    A, AL, AR, C = random_mixed_gauge(D, d, normalise=True)

    H = Hamiltonian({'ZZ':-1, 'X':0.2}).to_matrix()

    H2 = np.kron(H, H)

    energy = evaluateEnergy(AL, AR, C, H, debug=True)
    print(f'Single copy energy: {energy}')

    energy2 = evaluateEnergy(AL, AR, C, H2)
    print(f'Two copy energy: {energy2}')

    print('Trying vumps...')

    _ , _, _, energies = gs_vumps(H2, 4, 2, maxiter=100, strategy='polar')
    print(energies)
    plt.figure()
    plt.plot(energies, '--')
    plt.title('Energies')
    plt.show()

