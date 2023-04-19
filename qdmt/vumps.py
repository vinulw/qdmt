import numpy as np
from numpy import linalg as la
from ncon import ncon

from numpy.linalg import qr
from scipy.sparse.linalg import eigs

import matplotlib.pyplot as plt

def random_mixed_gauge(d, D):
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

    return AL, AR, C

def evaluateEnergy(AL, AR, C, h):
    '''
    Evaluate the expectation for energy <h> for a uMPS state in mixed canonical
    form. Assume that h is a two site operator for now.

    TODO: Extend this to an n-site operator
    '''
    h = h.reshape(2, 2, 2, 2)
    C = np.diag(C)

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

    δ = 1
    count = 0
    while δ > tol and maxiter > count:
        e = evaluateEnergy(AL, AR, C, h)

        h_shifted = h - e*np.eye(h.shape[0])

        LH = sumLeft(AL, h_shifted)
        LR = sumRight(AR, h_shifted)

        # TODO: Implementation still not complete
        count += 1

def sumLeft(AL, h, tol=1e-8):
    from scipy.sparse.linalg import bicgstab
    from numpy.linalg import solve
    D, d, _ = AL.shape
    h = h.reshape(d, d, d, d)

    Hl = ncon((AL, AL, h, AL.conj(), AL.conj()),
            ((1, 2, 4), (1, 3, 5), (3, 7, 2, 6), (4, 6, -1), (5, 7, -2)))


    ELL = ncon([AL, AL.conj()], ((-1, 1, -3), (-2, 1, -4)))
    ELL = ELL.reshape(D*D, D*D)

    e_left = largest_evec_left(ELL)
    e_right = largest_evec_right(ELL)

    # To check if the evecs are correct
    #ELLten = ELL.reshape(D, D, D, D)
    #e_left = e_left.reshape(D, D)
    #e_right = e_right.reshape(D, D)

    #out =  ncon([e_left, ELLten], ((1, 2), (1, 2, -1, -2)))
    #print(np.allclose(out, e_left)) # Largest eigenval = 1

    #out =  ncon([e_right, ELLten], ((1, 2), (-1, -2, 1, 2)))
    #print(np.allclose(out, e_right))

    P = np.outer(e_right, e_left)
    #Q = np.eye(D**2) - P # Not sure if this should wrap pseudo inverse

    E_psuedo = np.eye(D**2)  - (ELL - P)
    E_psuedoL = E_psuedo.conj().T

    Hl_dag = Hl.reshape(-1).conj().T
    # Suggested bicgstab in literature but solve works fast + more accurately for now
    # L, exitcode = bicgstab(E_psuedoL, Hl_dag, atol=1e-7)
    # print(exitcode)
    L = solve(E_psuedoL, Hl_dag)

    mapL = E_psuedoL.dot(L)
    # print('Checking dot product...')
    # print(np.allclose(mapL, E_psuedoL @ L))

    # print('Checking output of bicgstab...')
    # print(np.allclose(mapL, Hl_dag, atol=1e-5))

    # print('Norm diff')
    # print(np.linalg.norm(mapL - Hl_dag))

    Lh = L.conj().transpose().reshape(D, D)
    E_pseudo = E_psuedo.reshape(D, D, D, D)

    map_Lh = ncon([E_pseudo, Lh], ((1, 2, -1, -2), (1, 2)))

    print(np.allclose(map_Lh, Hl, atol=1e-5))
    # print(np.linalg.norm(map_Lh - Hl))

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

    R = solve(E_psuedo, Hr.reshape(-1)) # Replace with bicstag for large D

    Rh = R.reshape(D, D)
    E_psuedo = E_psuedo.reshape(D, D, D, D)

    map_Rh = ncon([E_psuedo, Rh], ((-1, -2, 1, 2), (1, 2)))
    print(np.allclose(map_Rh, Hr))

    return Rh

def largest_evec_left(E, l0 = None):
    '''
    Find leading eigenvector v of E such that vE = λv
    '''
    Eh = E.conj().T
    if l0 is not None:
        l0 = l0.conj().transpose()

    w, v = eigs(Eh, k=1, which='LM')

    e = v[:, 0]
    e = e.conj().transpose()

    return e

def largest_evec_right(E, r0 = None):
    '''
    Find leading eigenvector v of E such that Ev = λv
    '''
    w, v = eigs(E, k=1, which='LM', v0=r0)

    e = v[:, 0]

    return e

if __name__=="__main__":
    d = 2
    D = 4
    AL, AR, C = random_mixed_gauge(d, D)

    H = Hamiltonian({'ZZ':-1, 'X':0.2}).to_matrix()

    energy = evaluateEnergy(AL, AR, C, H)

    print(energy)

    print('Checking sum Left...')
    sumLeft(AL, H)

    print('Checking sum right...')
    sumRight(AR, H)
