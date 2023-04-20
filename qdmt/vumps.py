import numpy as np
from numpy import linalg as la
from ncon import ncon

from numpy.linalg import qr
from scipy.sparse.linalg import eigs
from scipy.linalg import polar
from scipy.sparse.linalg import bicgstab
from numpy.linalg import solve

import matplotlib.pyplot as plt
from tqdm import tqdm
from hamiltonian import Hamiltonian

def random_mixed_gauge(d, D, normalise=False):
    '''
    Generate a random mixed canonical uMPS state with bond dimension D and
    physical dimension d

    Returns
    -------
    AL :  Left orthonormal tensor representing A
    AR :  Right orthonormal tensor representing A
    C  :  Central tensor singular values, use np.diag(C) to get full matrix
    '''

    C = np.random.rand(D)
    C = C / la.norm(C) # Normalisation of C

    AL = np.random.rand(D*d, D) + 1j*np.random.rand(D*d, D)
    AR = np.random.rand(D,d*D) + 1j*np.random.rand(D, d*D)
    AL = (la.svd(AL, full_matrices=False)[0]).reshape(D, d, D)
    AR = (la.svd(AR, full_matrices=False)[2]).reshape(D, d, D)

    # Normalize the TM
    if normalise:
        TML = ncon([AL, AL.conj()], ((-1, 1, -3), (-2, 1, -4)))
        _, S, _ = la.svd(TML.reshape(D**2, D**2))
        AL = AL / np.sqrt(S[0])

        TMR = ncon([AR, AR.conj()], ((-1, 1, -3), (-2, 1, -4)))
        _, S, _ = la.svd(TMR.reshape(D**2, D**2))
        AR = AR / np.sqrt(S[0])

    return AL, AR, C

def normalise_A(A):
    '''
    Normalise A such that the transfer matrix of A has a leading eigenvalue of 1.
    '''
    D, _, _ = A.shape
    TM = ncon([A, A.conj()], ((-1, 1, -3), (-2, 1, -4))).reshape(D**2, D**2)

    _, S, _ = la.svd(TM)

    return A / np.sqrt(S[0])

def evaluateEnergy(AL, AR, C, h):
    '''
    Evaluate the expectation for energy <h> for a uMPS state in mixed canonical
    form. Assume that h is a two site operator for now.

    TODO: Extend this to an n-site operator
    '''
    h = h.reshape(2, 2, 2, 2)

    ACl = ncon([AL, C], ((-1, -2, 1), (1, -3)))
    ACr = ncon([C, AR], ((-1, 1), (1, -2, -3)))

    energyL = ncon([AL, AL.conj(), h, ACl, ACl.conj()],
                    ((1, 2, 3), (1, 4, 5), (4, 7, 2, 6), (3, 6, 8), (5, 7, 8)))

    energyR = ncon([ACr, ACr.conj(), h, AR, AR.conj()],
                    ((1, 2, 3), (1, 4, 5), (4, 7, 2, 6), (3, 6, 8), (5, 7, 8)))

    energy = 0.5*(energyL + energyR)

    return energy, energyL, energyR

def gs_vumps(h, d, D, tol=1e-5, maxiter=100):
    '''
    Perform vumps to optimise local hamiltonian h.
    '''
    from scipy.sparse.linalg import eigsh
    AL, AR, C = random_mixed_gauge(d, D)
    C = np.diag(C)
    h0 = h.copy()

    δ = 1
    count = 0
    energies = []
    error_δs = []
    error_ϵLs = []
    error_ϵRs = []
    while δ > tol and maxiter > count:
        e, eL, eR = evaluateEnergy(AL, AR, C, h0)
        energies.append(e)
        print(f'Current energy : {e}')

        h_shifted = h - e*np.eye(d**2)
        h_shiftedL = (h - eL*np.eye(d**2)).reshape(d, d, d, d)
        h_shiftedR = (h - eR*np.eye(d**2)).reshape(d, d, d, d)
        h_shifted = h_shifted.reshape(d, d, d, d)

        print('Shifted energy left...')
        ALc = ncon([AL, C], ((-1, -2, 1), (1, -3)))
        energyL = ncon([AL, AL.conj(), h_shiftedL, ALc, ALc.conj()],
                ((1, 2, 3), (1, 4, 5), (4, 7, 2, 6), (3, 6, 8), (5, 7, 8)))
        print(energyL)

        print('Shifted energy right...')
        ARc = ncon([C, AR], ((-1, 1), (1, -2, -3)))
        energyR = ncon([ARc, ARc.conj(), h_shiftedR, AR, AR.conj()],
                ((1, 2, 3), (1, 4, 5), (4, 7, 2, 6), (3, 6, 8), (5, 7, 8)))
        print(energyR)

        print('Shifted energy centre...')
        energyC = ncon([AL, C, AL.conj(), h.reshape(d, d, d, d), AR, C, AR.conj()],
                ((1, 2, 3), (3, 6), (1, 4, 5), (4, 10, 2, 8), (6, 8, 9), (5, 7), (7, 10, 9)))
        print(f'Before: {energyC}')
        h_shiftedC = (h - energyC*np.eye(d**2)).reshape(d, d, d, d)
        energyC = ncon([AL, C, AL.conj(), h_shiftedC, AR, C, AR.conj()],
                ((1, 2, 3), (3, 6), (1, 4, 5), (4, 10, 2, 8), (6, 8, 9), (5, 7), (7, 10, 9)))
        print(f'After: {energyC}')

        LH = sumLeft(AL, h_shiftedL)
        RH = sumRight(AR, h_shiftedR)


        # Create identity maps
        I = np.eye(D)
        Id = np.eye(d)

        # Construct C′
        C_map = ncon([AL, AL.conj(), h_shiftedC, AR, AR.conj()],
                       ((1, 2, -1), (1, 3, -3), (3, 6, 2, 4), (-2, 4, 5),
                           (-4, 6, 5)))
        C_map += ncon([LH, I], ((-1, -3), (-2, -4)))
        C_map += ncon([I, RH], ((-1, -3), (-2, -4)))

        C_map_reshaped = C_map.reshape(D**2, D**2)
        _, v = eigsh(C_map_reshaped, k=1, which='SA', v0=C.reshape(-1))

        C_prime = v[:, 0].reshape(D, D)

        # Construct Ac′
        AC_map = ncon([AL, AL.conj(), h_shiftedL, I],
                        ((1, 2, -1), (1, 3, -4), (3, -5, 2, -2), (-3, -6)))
        AC_map += ncon([I, h_shiftedR, AR, AR.conj()],
                         ((-1, -4), (-5, 3, -2, 1), (-3, 1, 2), (-6, 3, 2)))
        AC_map += ncon([LH, Id, I], ((-1, -4), (-2, -5), (-3, -6)))
        AC_map += ncon([I, Id, RH], ((-1, -4), (-2, -5), (-3, -6)))

        AC_map_reshaped = AC_map.reshape(d*D**2, d*D**2)

        AC = ncon([C, AR], ((-1, 1), (1, -2, -3)))
        _, v = eigsh(AC_map_reshaped, k=1, which='SA', v0=AC.reshape(-1))

        AC_prime = v[:, 0]
        AC_prime = AC_prime.reshape(D, d, D)
        # Not sure if this is strictly necessary
        AC_prime =  normalise_A(AC_prime)

        AL, AR, ϵL, ϵR = minAcC(AC_prime, C_prime, errors=True)

        # Change to diagonal gauge
        # TODO : Check if this is actually working correctly.
        # U, C_prime, V = la.svd(C_prime)
        # C_prime = np.diag(C_prime)
        # AL = ncon([U.conj().T, AL, U], ((-1, 1), (1, -2, 2), (2, -3)))
        # AR = ncon([V, AR, V.conj().T], ((-1, 1), (1, -2, 2), (2, -3)))
        # AC_prime = ncon([U.conj().T, AC_prime, U], ((-1, 1), (1, -2, 2), (2, -3)))

        C = C_prime.copy()
        AC = AC_prime.copy()

        # Check convergence
        H_AC = ncon([AC_map, AC], ((1, 2, 3, -1, -2, -3), (1, 2, 3)))
        H_C = ncon([C_map, C], ((1, 2, -1, -2), (1, 2)))
        AL_HC = ncon([AL, H_C], ((-1, -2, 1), (1, -3)))

        δ =  np.linalg.norm(H_AC - AL_HC)
        count += 1 # iteratre counter for maxiter

        e, _, _ = evaluateEnergy(AL, AR, C, h) # Calculate the final energy

        energies.append(e)
        error_δs.append(δ)
        error_ϵLs.append(ϵL)
        error_ϵRs.append(ϵR)

        print(f'Energy after opt: {e}')
        print('Errors: ')
        print(f'   δ: {δ}')
        print(f'   ϵL: {ϵL}')
        print(f'   ϵR: {ϵR}')

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

def sumLeft(AL, h, tol=1e-8):
    D, d, _ = AL.shape
    h = h.reshape(d, d, d, d)

    Hl = ncon((AL, AL, h, AL.conj(), AL.conj()),
            ((1, 2, 4), (1, 3, 5), (3, 7, 2, 6), (4, 6, -1), (5, 7, -2)))

    # Construct reduced transfer matrix
    ELL = ncon([AL, AL.conj()], ((-1, 1, -3), (-2, 1, -4)))
    ELL = ELL.reshape(D*D, D*D)
    U, S, V = la.svd(ELL)
    S[0] = 0 # Projecting out leading order term
    E_tilde = U @ np.diag(S) @ V

    # Setting up system of linear eq to solve for Lh
    E_psuedo = np.eye(D**2)  - E_tilde
    E_psuedoL = E_psuedo.conj().T
    Hl_dag = Hl.reshape(-1).conj()
    # Suggested bicgstab in literature but solve works fast + more accurately for now
    # L, exitcode = bicgstab(E_psuedoL, Hl_dag, atol=1e-7)
    # print(exitcode)
    L = solve(E_psuedoL, Hl_dag)
    Lh = L.conj().reshape(D, D)

    return Lh

def sumRight(AR, h, tol=1e-8):
    D, d, _ = AR.shape
    h = h.reshape(d, d, d, d)

    Hr = ncon((AR, AR.conj(), h, AR, AR.conj()),
              ((-1, 1, 3), (-2, 2, 4), (2, 6, 1, 5), (3, 5, 7), (4, 6, 7)))

    # Set up transfer matrix
    ELL = ncon([AR, AR.conj()], ((-1, 1, -3), (-2, 1, -4)))
    ELL = ELL.reshape(D*D, D*D)

    U, S, V = la.svd(ELL)
    S[0] = 0 # Projecting out leading order term
    E_tilde = U @ np.diag(S) @ V

    # Setting up system of linear eq to solve for Lh
    E_psuedo = np.eye(D**2)  - E_tilde
    R = solve(E_psuedo, Hr.reshape(-1)) # Replace with bicstag for large D
    Rh = R.reshape(D, D)

    return Rh

def minAcC(Ac, C, errors=False):
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
    AL, AR, C = random_mixed_gauge(d, D, normalise=True)
    C = np.diag(C)

    H = Hamiltonian({'ZZ':-1, 'X':0.2}).to_matrix()


    print('Checking sum Left...')
    sumLeft(AL, H)

    print('Checking sum right...')
    sumRight(AR, H)

    energy = evaluateEnergy(AL, AR, C, H)
    print('Trying vumps...')

    _ , _, _, energies = gs_vumps(H, 2, 4, maxiter=500)
    plt.figure()
    plt.plot(energies, 'x-')
    plt.title('Energies')
    plt.show()
