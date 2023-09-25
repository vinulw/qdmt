import numpy as np
from numpy.linalg import svd
import vumps as v
from vumpt_tools import mixedCanonical
from ncon import ncon
from tqdm import tqdm
from scipy.optimize import minimize
import os

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
    return np.real(np.trace(dist @ dist.conj().T))

def cost_trace_distance(A, rho_target):
    D = 2
    d = 2
    A = A.reshape(D, d, D)
    A = v.normalizeMPS(A)
    Al, Ac, Ar, C = mixedCanonical(A)

    rho = mixed_state_to_two_site_rho(Al, C, Ar)
    rho_mat = rho.reshape(d**2, d**2)

    return trace_dist(rho_target, rho_mat)

def get_spectrum(ρ):
    ρ_ = ρ.reshape(4, 4)

    # _, s, _ = svd(ρ_)

    s, _ = np.linalg.eig(ρ_)
    return s

def main_opt_vumps():
    D = 2
    d = 2

    N = 100
    print('Collecting dist data...')

    tols = [1e-6, 1e-8, 1e-10, 1e-12]
    for t in tols:
        print(f'Gathering tol: {t}')
        t_dists = np.zeros(N)
        messages = []
        for i in tqdm(range(N)):
            A = v.createMPS(D, d)
            A = v.normalizeMPS(A)

            Al, Ac, Ar, C = mixedCanonical(A)

            rho = mixed_state_to_two_site_rho(Al, C, Ar)
            rho_mat = rho.reshape(d**2, d**2)


            I = np.eye(4).reshape(2, 2, 2, 2)
            rho =  I - rho
            rho = rho / np.linalg.norm(rho)

            maxiter = 100
            errors = np.zeros(maxiter)
            energies = np.zeros(maxiter)
            def callback(count, tensors, delta, energy):
                errors[count-1] = delta
                energies[count-1] = energy

            Al, Ac, Ar, C, message = v.vumps(rho, D, d, tol=t, tolFactor=1e-2,
                                             verbose=False, message=True,
                                             maxiter=maxiter, callback=callback, M_opt=rho_mat)

            # print('Errors: ')
            # print(errors)
            # print('Energies: ')
            # print(energies)
            # folder = 'data/state_optimise'

            # fname = os.path.join(folder, 'test_errors.npy')
            # np.save(fname, errors)

            # fname = os.path.join(folder, 'test_energies.npy')
            # np.save(fname, energies)
            # assert()

            rho_opt = mixed_state_to_two_site_rho(Al, C, Ar)
            rho_opt_mat = rho_opt.reshape(d**2, d**2)

            t_dists[i] = trace_dist(rho_mat, rho_opt_mat)
            print(t_dists[i])
            print('Ideal trace distance...')
            print(trace_dist(rho_mat, rho_mat))
            assert()
            messages.append(message)

        folder = 'data/state_optimise'
        tolstr = str(t)[-2:]
        fname = os.path.join(folder, f'trace_opt_state_tol{tolstr}.npy')
        print('Saving data: ', fname)
        np.save(fname, t_dists)
        fname = os.path.join(folder, f'messages_opt_state_tol{tolstr}.npy')
        print('Saving data: ', fname)
        np.save(fname, messages)
        print()


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

