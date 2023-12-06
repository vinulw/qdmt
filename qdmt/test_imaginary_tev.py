import numpy as np
import vumps as v
from ncon import ncon
from ground_state import Hamiltonian
from scipy.linalg import expm
import matplotlib.pyplot as plt

from vumpt_tools import mixedCanonical

def evolve_2_site(Al, C, Ar, Ueven, Uodd):
    Uodd_dag = Uodd.conj().T.reshape(2, 2, 2, 2)
    Ueven_dag = Ueven.conj().T.reshape(2, 2, 2, 2)
    Uodd = Uodd.reshape(2, 2, 2, 2)
    Ueven = Ueven.reshape(2, 2, 2, 2)

    # Apply odd + even evolution

    tensors = [Al, Al, C, Ar, Ar, Uodd, Uodd, Ueven]
    tensor_dag = [Al.conj(), Al.conj(), C.conj(), Ar.conj(), Ar.conj(),
                  Uodd_dag, Uodd_dag, Ueven_dag]
    edges = [
            (-1, 1, 2), (2, 3, 4),  # Al, Al
            (4, 6),  # C
            (6, 7, 9), (9, 10, -6),  # Ar, Ar
            (-2, 5, 1, 3), (8, -5, 7, 10),  # Uodd, Uodd
            (-3, -4, 5, 8)  # Ueven
            ]

    state_tev = ncon(tensors, edges)
    state_dag_tev = ncon(tensor_dag, edges)

    # Trace out
    tensors = [state_tev, state_dag_tev]
    edges = [
            (1, 2, -1, -2, 5, 6),
            (1, 2, -3, -4, 5, 6)
            ]
    ρt = ncon(tensors, edges)
    return ρt



if __name__=="__main__":
    # Load the exact data
    g_exact, e_exact = np.loadtxt('../data/exact_gs_energy.csv', delimiter=',')

    g = g_exact[0]
    e_analytic = e_exact[0]

    print('Current g: ', g)
    print('Exact energy: ', e_analytic)

    # Generate a random density matrix
    print('Generating random ρ...')
    d = 2
    D = 2
    A = v.createMPS(D, d)
    print(A.shape)
    A = v.normalizeMPS(A)
    Al, Ac, Ar, C = mixedCanonical(A)

    print('Generating U...')
    H = Hamiltonian({'ZZ':-1, 'X': g}).to_matrix()
    h = H.reshape(2, 2, 2, 2)
    dt = 0.01

    U = expm(-1*H*dt)

    print('Evolving ρ...')
    ρt = evolve_2_site(Al, C, Ar, U, U)

    # Test properties of ρt

    # Hermitian?
    M = ρt.reshape(d**2, d**2)

    print('ρt is Hermitian? ',  np.allclose(M, M.conj().T))

    # Brute force optimisation
    from scipy.optimize import minimize
    print('Attempting brute force...')
    def cost_function(A, ρ):
        A = A.reshape(2, 2, 2)
        AL, AC, AR, C = v.mixedCanonical(A)
        E = np.real(v.expValNMixed(ρ, AC, AR))
        return -1*E

    res = minimize(cost_function, A.flatten(), args=(ρt,))
    print('Final expVal: ', res.fun)

    print('Attempting brute force time evolution...')
    nsteps = 200
    ts = [None]*nsteps
    Es = [None]*nsteps
    for i in range(nsteps):
        ρt = evolve_2_site(Al, C, Ar, U, U)

        res = minimize(cost_function, Al.flatten(), args=(ρt,))
        Aprime = res.x.reshape(2, 2, 2)
        Aprime = v.normalizeMPS(Aprime)
        Al, Ac, Ar, C = v.mixedCanonical(Aprime)
        E = np.real(v.expValNMixed(h, Ac, Ar))

        ts[i] = dt*(i+1)
        Es[i] = E
        print('Time step: ', ts[i])
        print('Energy: ', E)
        print('δE: ', E - e_analytic )
        print()

    plt.plot(ts, Es)
    plt.hline(e_analytic)
    plt.show()

    assert False

    # This is not working yet
    print('Applying vumps...')
    Al, Ac, Ar, C = v.vumps(ρt, D, d, tol=1e-8, tolFactor=1e-2, verbose=True, maxiter=200)
#
    print('Applied vumps...')
