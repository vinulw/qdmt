import numpy as np
from numpy import linalg as la
from ncon import ncon

from numpy.linalg import qr
from scipy.sparse.linalg import eigs
from scipy.linalg import polar

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

def evaluateEnergy(AL, AR, C, h):
    '''
    Evaluate the expectation for energy <h> for a uMPS state in mixed canonical
    form. Assume that h is a two site operator for now.

    TODO: Extend this to an n-site operator
    '''
    h = h.reshape(2, 2, 2, 2)

    AC_L = ncon([AL, C], ((-1, -2, 1), (1, -3)))

    energy = ncon([AC_L, AC_L.conj(), h, AR, AR.conj()],
                  ((1, 2, 4), (1, 3, 5), (3, 7, 2, 6), (4, 6, 8), (5, 7, 8)),
                  (1, 2, 3, 4, 5, 6, 7, 8)) # Specify contraction order

    return energy

def gs_vumps(h, d, D, tol=1e-5, maxiter=100):
    '''
    Perform vumps to optimise local hamiltonian h.
    '''
    AL, AR, C = random_mixed_gauge(d, D)
    C = np.diag(C)

    δ = 1
    count = 0
    energies = []
    while δ > tol and maxiter > count:
        e = evaluateEnergy(AL, AR, C, h)
        energies.append(e)
        print(f'Current energy : {e}')

        h_shifted = h - e*np.eye(d**2)
        h_shifted = h_shifted.reshape(d, d, d, d)

        LH = sumLeft(AL, h_shifted)
        RH = sumRight(AR, h_shifted)

        # Construct Ac′
        AC = ncon([C, AR], ((-1, 1), (1, -2, -3)))
        I = np.eye(D) # Create identity map
        Id = np.eye(d)

        AC_map = ncon([AL, AL.conj(), h_shifted, I],
                        ((1, 2, -1), (1, 3, -4), (3, -5, 2, -2), (-3, -6)))
        AC_map += ncon([I, h_shifted, AR, AR.conj()],
                         ((-1, -4), (-5, 3, -2, 1), (-3, 1, 2), (-6, 3, 2)))
        AC_map += ncon([LH, Id, I], ((-1, -4), (-2, -5), (-3, -6)))
        AC_map += ncon([I, Id, RH], ((-1, -4), (-2, -5), (-3, -6)))

        AC_map_reshaped = AC_map.reshape(d*D**2, d*D**2)

        _, v = eigs(AC_map_reshaped, k=1, which='LM', v0=AC.reshape(-1))

        AC_prime = v[:, 0]
        AC_prime = AC_prime.reshape(D, d, D)

        # Construct C′
        C_map = ncon([AL, AL.conj(), h_shifted, AR, AR.conj()],
                       ((1, 2, -1), (1, 3, -3), (3, 6, 2, 4), (-2, 4, 5),
                           (-4, 6, 5)))
        C_map += ncon([LH, I], ((-1, -3), (-2, -4)))
        C_map += ncon([I, RH], ((-1, -3), (-2, -4)))

        C_map_reshaped = C_map.reshape(D**2, D**2)
        _, v = eigs(C_map_reshaped, k=1, which='LM', v0=C.reshape(-1))

        C_prime = v[:, 0].reshape(D, D)

        AL, AR, AC, C = minAcC(AC_prime, C_prime)

        # Check convergence
        H_AC = ncon([AC_map, AC], ((1, 2, 3, -1, -2, -3), (1, 2, 3)))
        H_C = ncon([C_map, C], ((1, 2, -1, -2), (1, 2)))
        AL_HC = ncon([AL, H_C], ((-1, -2, 1), (1, -3)))

        δ =  np.linalg.norm(H_AC - AL_HC)
        count += 1 # iteratre counter for maxiter

        e = evaluateEnergy(AL, AR, C, h) # Calculate the final energy

        print(f'Energy after opt: {e}')
        breakpoint()


    e = evaluateEnergy(AL, AR, C, h) # Calculate the final energy
    energies.append(e)
    return AL, AR, C, energies

def sumLeft(AL, h, tol=1e-8):
    from scipy.sparse.linalg import bicgstab
    from numpy.linalg import solve
    D, d, _ = AL.shape
    h = h.reshape(d, d, d, d)

    Hl = ncon((AL, AL, h, AL.conj(), AL.conj()),
            ((1, 2, 4), (1, 3, 5), (3, 7, 2, 6), (4, 6, -1), (5, 7, -2)))


    ELL = ncon([AL, AL.conj()], ((-1, 1, -3), (-2, 1, -4)))
    ELL = ELL.reshape(D*D, D*D)
    U, S, V = la.svd(ELL)

    print('Singular values before...')
    print(S)

    print('Projecting out leading order...')
    S[0] = 0
    E_tilde = U @ np.diag(S) @ V

    #e_left = largest_evec_left(ELL).reshape(D, D)
    #e_right = largest_evec_right(ELL).reshape(D, D)

    #P = ncon([e_right, e_left], ((-1, -2), (-3, -4))).reshape(D**2, D**2)
    #Q = np.eye(D**2) - P # Not sure if this should wrap pseudo inverse

    #E_tilde = (ELL - P) # remove leading order term
    #E_tilde = Q@ELL
    print('Singular Values after...')
    _, S, _ = la.svd(E_tilde)
    print(S)
    E_psuedo = np.eye(D**2)  - E_tilde


    E_pinv = la.inv(E_psuedo)

    print('Verifying inverse...')
    print(np.allclose(np.eye(D**2), E_pinv @ E_psuedo))

    errors = []

    E_tilde_curr = np.copy(E_tilde)
    E_tilde_sum = np.eye(D**2)
    n = 100
    print('Verifying that repetition approaches pseudo inverse...')
    for i in tqdm(range(n), total=n):
        # E_tilde_curr = ncon([E_tilde_curr, E_tilde], ((-1, 1), (1, -2)))
        E_tilde_sum = E_tilde_sum + E_tilde_curr
        E_tilde_curr = E_tilde @ E_tilde_curr

        errors.append(la.norm(E_pinv - E_tilde_sum))

        # print(f'Current error: {errors[-1]}')
    plt.title('Error in E_tilde')
    plt.plot(errors)

    E_psuedoL = E_psuedo.conj().T

    Hl_dag = Hl.reshape(-1).conj()
    # Suggested bicgstab in literature but solve works fast + more accurately for now
    # L, exitcode = bicgstab(E_psuedoL, Hl_dag, atol=1e-7)
    # print(exitcode)
    L = solve(E_psuedoL, Hl_dag)

    mapL = E_psuedoL.dot(L)
    print('Checking dot product...')
    print(np.allclose(mapL, E_psuedoL @ L))

    # print('Checking output of bicgstab...')
    # print(np.allclose(mapL, Hl_dag, atol=1e-5))

    print('Norm diff')
    print(np.linalg.norm(mapL - Hl_dag))

    Lh = L.conj().reshape(D, D)
    E_pseudo = E_psuedo.reshape(D, D, D, D)

    map_Lh = ncon([E_pseudo, Lh], ((1, 2, -1, -2), (1, 2)))

    print(np.allclose(map_Lh, Hl, atol=1e-5))
    print(np.linalg.norm(map_Lh - Hl))

    Hl_Einv = np.dot(Hl.reshape(-1),  E_pinv)
    Hl_Einv = Hl_Einv.reshape(D, D)

    print('Checking Lh == Hl.E_pinv')
    print(np.linalg.norm(Hl_Einv - Lh))


    plt.show()
    assert()

    return Lh

def sumRight(AR, h, tol=1e-8):
    from numpy.linalg import solve
    D, d, _ = AR.shape
    h = h.reshape(d, d, d, d)

    Hr = ncon((AR, AR.conj(), h, AR, AR.conj()),
              ((-1, 1, 3), (-2, 2, 4), (2, 6, 1, 5), (3, 5, 7), (4, 6, 7)))


    ELL = ncon([AR, AR.conj()], ((-1, 1, -3), (-2, 1, -4)))
    ELL = ELL.reshape(D*D, D*D)

    e_left = largest_evec_left(ELL)
    e_right = largest_evec_right(ELL)

    P = np.outer(e_right, e_left)
    E_psuedo = np.eye(D**2)  - (ELL - P)
    # E_psuedo = np.eye(D**2)  - (ELL)

    R = solve(E_psuedo, Hr.reshape(-1)) # Replace with bicstag for large D

    Rh = R.reshape(D, D)
    E_psuedo = E_psuedo.reshape(D, D, D, D)

    map_Rh = ncon([E_psuedo, Rh], ((-1, -2, 1, 2), (1, 2)))
    # print(np.allclose(map_Rh, Hr))

    return Rh

def minAcC(Ac, C):
    Ul_Ac, Pl_Ac = polar(Ac.reshape(D*d, D), side='left')
    Ul_C, Pl_C = polar(C, side='left')

    Al = Ul_Ac @ (Ul_C.conj().T)
    Al = Al.reshape(D, d, D)

    Ur_Ac, Pr_Ac = polar(Ac.reshape(D*d, D), side='right')
    Ur_C, Pr_C = polar(C, side='right')

    Ar = Ur_Ac @ (Ur_C.conj().T)
    Ar = Ar.reshape(D, d, D)

    C_updated = Pl_C @ Ul_C
    Ac_updated = (Pl_Ac @ Ul_Ac).reshape(D, d, D)
    return Al, Ar, Ac_updated, C_updated

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

    assert()

    print('Checking sum right...')
    sumRight(AR, H)

    energy = evaluateEnergy(AL, AR, C, H)
    print('Trying vumps...')

    _ , _, _, energies = gs_vumps(H, 2, 4, maxiter=500)
    plt.plot(energies, 'x-')
    plt.show()
