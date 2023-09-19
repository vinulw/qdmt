import numpy as np
import vumps as v
from vumpt_tools import mixedCanonical
from ncon import ncon
from tqdm import tqdm
from scipy.optimize import minimize

def mixed_state_to_two_site_rho(Al, C, Ar):
    edges = [
            (1, -1, 2), (1, -3, 3),
            (2, 4), (3, 5),
            (4, -2, 6), (5, -4, 6)
            ]
    tensors = [Al, Al.conj(),
               C, C.conj(),
               Ar, Ar.conj()]

    return ncon(tensors, edges)

def trace_dist(A, B):
    assert A.shape == B.shape
    assert len(A.shape) == 2

    dist = A - B
    return np.real(np.trace(dist * dist.conj().T))

def cost_trace_distance(A, rho_target):
    D = 2
    d = 2
    A = A.reshape(D, d, D)
    A = v.normalizeMPS(A)
    Al, Ac, Ar, C = mixedCanonical(A)

    rho = mixed_state_to_two_site_rho(Al, C, Ar)
    rho_mat = rho.reshape(d**2, d**2)

    return trace_dist(rho_target, rho_mat)


def main_opt_vumps():
    D = 2
    d = 2

    N = 100
    t_dists = np.zeros(N)
    print('Collecting dist data...')
    for i in tqdm(range(N)):
        A = v.createMPS(D, d)
        A = v.normalizeMPS(A)

        Al, Ac, Ar, C = mixedCanonical(A)

        rho = mixed_state_to_two_site_rho(Al, C, Ar)
        rho_mat = rho.reshape(d**2, d**2)

        I = np.eye(4).reshape(2, 2, 2, 2)
        rho = rho - I

        Al, Ac, Ar, C = v.vumps(rho, D, d, tol=1e-8, tolFactor=1e-2, verbose=False)

        rho_opt = mixed_state_to_two_site_rho(Al, C, Ar)
        rho_opt_mat = rho_opt.reshape(d**2, d**2)

        t_dists[i] = trace_dist(rho_mat, rho_opt_mat)

    fname = 'data/tdist_optimising_state_minus_eye.npy'
    print('Saving data: ', fname)
    np.save(fname, t_dists)

def main_opt_minimize():
    D = 2
    d = 2

    N = 100
    t_dists = np.zeros(N)
    print('Collecting dist data...')
    for i in tqdm(range(N)):
        A = v.createMPS(D, d)
        A = v.normalizeMPS(A)

        Al, Ac, Ar, C = mixedCanonical(A)

        rho = mixed_state_to_two_site_rho(Al, C, Ar)
        rho_mat = rho.reshape(d**2, d**2)

        x0 = np.random.randn(D**2*d)
        res = minimize(cost_trace_distance, x0, args=(rho_mat,))
        A = res.x
        A = A.reshape(D, d, D)
        A = v.normalizeMPS(A)
        Al, Ac, Ar, C = mixedCanonical(A)

        rho_opt = mixed_state_to_two_site_rho(Al, C, Ar)
        rho_opt_mat = rho_opt.reshape(d**2, d**2)

        t_dists[i] = trace_dist(rho_mat, rho_opt_mat)

    fname = 'data/tdist_optimising_state_minimize.npy'
    print('Saving data: ', fname)
    np.save(fname, t_dists)



if __name__=="__main__":
    main_opt_vumps()

