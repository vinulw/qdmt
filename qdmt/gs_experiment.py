from circuits import StateAnsatzXZ
from environment import generate_environment_unitary
from dmt_overlap import partial_density_matrix
import cirq
from ncon import ncon
import numpy as np
from ground_state import Hamiltonian

from scipy.optimize import minimize

def rho_theta(θ, N, d=2, D=2, ignored_indices=None, Ansatz=StateAnsatzXZ):
    '''
    Generate density matrix from params and Ansatz
    '''
    from environment import generate_state_transferMatrix, generate_right_environment
    U = cirq.unitary(StateAnsatzXZ(θ))

    zeroState = np.eye(d)[0, :]

    A = U.T.reshape(d, D, d, D)
    A = A.transpose(2, 3, 0, 1)
    A = ncon([A, zeroState], ((-1, -2, 1, -3), (1,))) # (Dl, σ, Dr)

    TM = generate_state_transferMatrix(A.transpose(1, 0, 2))
    RA = generate_right_environment(TM)
    RA = RA.reshape(D, D)
    norm = ncon([RA,], ((1, 1),))
    RA = RA / norm

    #RA = RA / np.linalg.norm(RA)

    # Generate the environment
    #s = np.eye(D**2)[0]
    #RUA = generate_environment_unitary(A, D=D)
    #RA = RUA @ s
    #RA = RA.reshape(D, D)


    overlap = ncon([A, A.conj(), RA], ((1, 2, 3), (1, 2, 4), (4, 3)))
    # print('Explicit overlap' , overlap)

    #RA = generate_right_environment()

    if ignored_indices is None:
        ignored_indices = range(N)

    return partial_density_matrix(A, RA, N=N, ignored_indices=ignored_indices, halfEnv=False)



def two_site_energy(T, H):
    from scipy.linalg import eig
    T_tm = tensor_to_transfer_matrix(T)
    _, l, r = eig(T_tm, left=True, right=True)
    l = l[:, 0]
    r = r[:, 0]

    n = np.dot(l, r)
    l = l / np.sqrt(n)
    r = r / np.sqrt(n)

    two_site_tm = two_site_transfer_matrix(T, H)

    energy = ncon([l, two_site_tm, r], ((1,), (1, 2), (2,)))

    return np.real(energy) #np.abs(right_fixed_point(two_site_tm)[0])

def two_site_transfer_matrix(T, H):
    i, _, k = T.shape
    TT = ncon([T, T, H, np.conj(T), np.conj(T)],
              ((-1,  1, 2), (2, 3, -3), (4, 5, 1, 3), (-2, 4, 6), (6, 5, -4))
              )
    return TT.reshape(2*i, 2*k)

def tensor_to_transfer_matrix(T):
    i, _, j = T.shape
    TM = ncon([T, np.conj(T)], ((-1, 1, -3), (-2, 1, -4)))
    return TM.reshape(2*i, 2*j)

def two_site_objective_function(θ, H):
    from classical import unitary_to_tensor_left

    U = cirq.unitary(StateAnsatzXZ(θ))
    T = unitary_to_tensor_left(U)

    E = two_site_energy(T, H)

    return E


if __name__=="__main__":
    θ = np.random.rand(8)
    d = 2
    D = 2
    N = 2
    g = 1.5

    ρ = rho_theta(θ, N, d, D)

    print('Norm of ρ')
    print(ncon([ρ,], ((1, 1, 2, 2),)))

    H = Hamiltonian({'ZZ':-1, 'X': 1.5}).to_matrix()

    H = H.reshape(2, 2, 2, 2)

    def rho_objective_function(θ, H):
        d = 2
        D = 2
        N = 2

        ρ = rho_theta(θ, N, d, D)
        exp = ncon([H, ρ], ((1, 2, 3, 4), (3, 1, 4, 2)))
        return np.real(exp)

    # Generate A to compare expectation value to.
    U = cirq.unitary(StateAnsatzXZ(θ))
    zeroState = np.eye(d)[0, :]
    A = U.T.reshape(d, D, d, D)
    A = A.transpose(2, 3, 0, 1)
    A = ncon([A, zeroState], ((-1, -2, 1, -3), (1,))) # (Dl, σ, Dr)

    obj_f = lambda x: two_site_objective_function(x, H)

    res = minimize(obj_f, θ)

    print('Final energy ψ : ', res)

    rho_obj_f = lambda x: rho_objective_function(x, H)

    res = minimize(rho_obj_f, θ)

    print('Final energy ρ : ', res)



