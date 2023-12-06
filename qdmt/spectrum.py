import numpy as np
from numpy.linalg import svd
from test_imaginary_tev import evolve_2_site
import vumps as v
from vumpt_tools import mixedCanonical
from ground_state import Hamiltonian
from scipy.linalg import expm
from hamiltonian import TransverseIsing

def generate_random_spectrum(U, d=2, D=2):
    A = v.createMPS(D, d)
    A = v.normalizeMPS(A)
    Al, Ac, Ar, C = mixedCanonical(A)

    ρt = evolve_2_site(Al, C, Ar, U, U)
    ρt = ρt.reshape(4, 4)

    _, s, _ = svd(ρt)

    return s

def main_density():
    H = Hamiltonian({'ZZ':-1, 'X': g}).to_matrix()
    g = 0.2
    h = H.reshape(2, 2, 2, 2)
    dt = 0.01

    #U = expm(-1*H*dt)
    U = np.eye(4)
    U = U.reshape(2, 2, 2, 2)

    N = 100
    svals = np.zeros((N, 4))

    for i in range(N):
        svals[i] = generate_random_spectrum(U)

    np.save('singular_vals_no_evolution.npy', svals)
    print(svals)

def main_hamiltonian():
    N = 100
    g_range = np.linspace(0.1, 1.6, N)

    svals = np.zeros((N, 4))
    for i, g in enumerate(g_range):
        h = TransverseIsing(1, g, 2)

        _, s, _ = svd(h)
        svals[i] = s

    np.save('data/singular_vals_ham.npy', svals)
    print(svals)

if __name__=="__main__":
    main_hamiltonian()




