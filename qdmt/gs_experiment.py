from circuits import StateAnsatzXZ
from environment import generate_environment_unitary
from dmt_overlap import partial_density_matrix
import cirq
from ncon import ncon
import numpy as np
from ground_state import Hamiltonian
from vumps import vumps, generate_ρ, expValNMixed

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

def evolve_ρ(ρ, U):
    '''
    Evole ρ using ρ(t + dt) = U ρ U^\dagger
    '''
    m = len(ρ.shape) // 2
    n = len(U.shape) // 2

    ρ_curr = np.copy(ρ)

    assert m % n == 0, 'U and ρ dimensions do not match'

    N = m // n
    len_i = len(ρ.shape)

    curr_i = 1
    ρ_con = list(range(-1, -len_i - 1, -1))
    U_cons = [None]*N
    U_dag_cons = [None]*N
    for i in range(N):
        U_con = (-curr_i, -curr_i - 2, curr_i, curr_i + 2)
        U_dag_con = (curr_i + 1, curr_i + 3, -curr_i - 1, -curr_i - 3)

        ρ_con[curr_i - 1] = curr_i
        ρ_con[curr_i] = curr_i + 1
        ρ_con[curr_i + 1] = curr_i + 2
        ρ_con[curr_i + 2] = curr_i + 3

        U_cons[i] = U_con
        U_dag_cons[i] = U_dag_con


        curr_i += 4

    arrs = [U] * N + [U.conj()] * N + [ρ_curr]
    cons = U_cons + U_dag_cons + [ρ_con]

    ρ_curr = ncon(arrs, cons)
    return ρ_curr

def evolve_ρ_trotter(ρ, U):
    '''
    Evolve ρ using the first order trotterisation of a single site
    '''
    N = len(ρ.shape) // 2
    assert N == 4 , 'For now the implementation only handles a single iteration of the transfer matrix'
    ρ_curr = np.copy(ρ)

    len_i = len(ρ.shape)

    # Apply odd layers
    Nodd = N // 2
    ρ_con = list(range(-1, -len_i - 1, -1))
    U_cons = [None] * Nodd
    U_dag_cons = [None] * Nodd
    count = 0
    for i in range(1, N*2, 4):
        U_con = (-i, -i-2, i, i+2)
        U_dag_con = (i+1, i+3, -i-1, -i-3)

        ρ_con[i - 1] = i
        ρ_con[i] = i + 1
        ρ_con[i + 1] = i + 2
        ρ_con[i + 2] = i + 3

        U_cons[count] = U_con
        U_dag_cons[count] = U_dag_con
        count+=1

    #print(ρ_con)
    #print(U_cons)
    #print(U_dag_cons)
    arrs = [U] * Nodd + [U.conj()] * Nodd + [ρ_curr]
    cons = U_cons + U_dag_cons + [ρ_con]
    ρ_curr = ncon(arrs, cons)

    # Apply even layers
    Neven = (N-1) // 2
    ρ_con = list(range(-1, -len_i - 1, -1))
    U_cons = [None] * Neven
    U_dag_cons = [None] * Neven
    count = 0
    for i in range(3, N*2-1, 4):
        U_con = (-i, -i-2, i, i+2)
        U_dag_con = (i+1, i+3, -i-1, -i-3)

        ρ_con[i - 1] = i
        ρ_con[i] = i + 1
        ρ_con[i + 1] = i + 2
        ρ_con[i + 2] = i + 3

        U_cons[count] = U_con
        U_dag_cons[count] = U_dag_con
        count+=1

    #print(ρ_con)
    #print(U_cons)
    #print(U_dag_cons)
    arrs = [U] * Neven + [U.conj()] * Neven + [ρ_curr]
    cons = U_cons + U_dag_cons + [ρ_con]
    ρ_curr = ncon(arrs, cons)

    # Contract the corner tensors
    ρ_con = list(range(-1, -len_i - 1, -1))
    ρ_con[0] = 1
    ρ_con[1] = 1
    ρ_con[-1] = 2
    ρ_con[-2] = 2

    ρ_curr = ncon([ρ_curr], [ρ_con])
    return ρ_curr



def evolve_ρ_sequential(ρ, U):
    '''
    Evole ρ using ρ(t + dt) = U ρ U^\dagger

    Do this sequentially so as to avoid resource constraints.
    '''
    m = len(ρ.shape) // 2
    n = len(U.shape) // 2

    ρ_curr = np.copy(ρ)

    assert m % n == 0, 'U and ρ dimensions do not match'

    N = m // n
    len_i = len(ρ.shape)

    curr_i = 1
    for _ in range(N):
        U_con = (-curr_i, -curr_i - 2, curr_i, curr_i + 2)
        U_dag_con = (curr_i + 1, curr_i + 3, -curr_i - 1, -curr_i - 3)

        ρ_con = list(range(-1, -len_i - 1, -1))
        ρ_con[curr_i - 1] = curr_i
        ρ_con[curr_i] = curr_i + 1
        ρ_con[curr_i + 1] = curr_i + 2
        ρ_con[curr_i + 2] = curr_i + 3

        ρ_curr = ncon([U, ρ_curr, U.conj()], (U_con, ρ_con, U_dag_con))

        curr_i += 4

    return ρ_curr

def normalise_ρ(ρ):
    len_ρ = len(ρ.shape) // 2
    con = list(range(1, len_ρ + 1))
    con = con + con

    return ρ / np.real(ncon((ρ, ), (con, )))


if __name__=="__main__":
    from datetime import datetime
    θ = np.random.rand(8)
    d = 2
    D = 2
    N = 2
    g = 1.5
    save = False

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
        # ρ = normalise_ρ(ρ)
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
    exact_energy = res.fun

    print('Final energy ψ : ', res.fun)
    print(res.x)

    rho_obj_f = lambda x: rho_objective_function(x, H)

    res = minimize(rho_obj_f, θ)
    θgs = res.x

    print('Final energy ρ : ', res.fun)
    print(θgs)

    if save:
        now = datetime.now()
        now = now.strftime("%d%m%Y%H%M%S")
        gs_fname = f'gs_{g}_{now}'
        print(f'Saving gs file as : {gs_fname}.npy')

        np.save(gs_fname, θgs)

    ####
    # Attempting imaginary time evolution groundstate search
    ####
    from scipy.linalg import expm
    from dmt_overlap import trace_distance
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    print('Performing imaginary time evolution ...')

    dt = 0.01
    max_t = 1.0
    ts = np.arange(0, max_t, dt)

    d = 2
    D = 2
    N = 4
    Ngs = 2

    H = Hamiltonian({'ZZ':-1, 'X': 1.5}).to_matrix()
    U = expm(-1*H*dt*2.0)
    U = U.reshape(2, 2, 2, 2)
    H = H.reshape(2, 2, 2, 2)
    θ0 = np.random.rand(8)

    def objective_func_trace(θdt, ρ0dt):
        N = len(ρ0dt.shape) // 2
        # Commented out for 4 site
        #N = 4
        ρt = rho_theta(θdt, N, d, D)
        # Comment below for 4 site
        #ρt = ncon([ρt], ((1, 1, -1, -2, -3, -4, 2, 2)))
        ρt = normalise_ρ(ρt)
        cost = np.abs(trace_distance(ρt, ρ0dt))
        return cost

    ρ0 = rho_theta(θ0, N, d, D)
    print('Made initial ρ')

    ρ_curr = np.copy(ρ0)
    θ_curr = np.copy(θ0)
    ρ_len = len(ρ_curr.shape) // 2
    norm_con = list(range(1, ρ_len + 1))
    norm_con = norm_con + norm_con

    energies = np.zeros(ts.shape[0])
    energies[0] = rho_objective_function(θ0, H)

    trace_dists = np.zeros(ts.shape[0] - 1)

    energies_exact = np.zeros(ts.shape[0])
    energies_exact[0] = energies[0]

    ρgs = rho_theta(θgs, Ngs, d, D)
    # ρgs = normalise_ρ(ρgs)
    #ρgs = ρgs / np.real(ncon((ρgs, ), (norm_con, )))
    gs_energy = ncon([H, ρgs], ((1, 2, 3, 4), (3, 1, 4, 2)))
    gs_energy = np.real(gs_energy)
    print('Target gs energy: ', gs_energy)
    #print('GS norm: ', np.real(ncon((ρgs, ), (norm_con, ))))


    for i, t in tqdm(enumerate(ts[1:]), total=len(ts[1:])):
        ρcurrdt = evolve_ρ_trotter(ρ_curr, U)
        ρcurrdt = normalise_ρ(ρcurrdt)

        Al, Ac, Ar, C = vumps(ρcurrdt, D, d, tol=1e-8, tolFactor=1e-2, verbose=False)
        ρ_curr = generate_ρ(Al, Ar, C, N)
        ρgs = generate_ρ(Al, Ar, C, Ngs)

        energy = ncon([H, ρgs], ((1, 2, 3, 4), (3, 1, 4, 2)))
        energies[i+1] = np.real(energy)

        # res = minimize(objective_func_trace, θ_curr, args=(ρcurrdt))

        # θ_curr = res.x
        # ρ_curr = rho_theta(θ_curr, N, d, D)

        # energies[i+1] = rho_objective_function(θ_curr, H)
        # trace_dists[i] = res.fun

        # if res.success is False:
        #     tqdm.write(f"Step: {i}")
        #     tqdm.write(f"   Obj func : {res.fun}")
        #     tqdm.write(f"   Message: {res.message}")

    print('Final energy : ', energies[-1])
    print(θ_curr)
    plt.figure()
    plt.title('Energies')
    plt.plot(ts, energies)
    fig, ax = plt.subplots()
    plt.title('Error in energy')
    ax.plot(ts, np.abs(energies - gs_energy))
    ax.set_yscale('log')
    #plt.figure()
    #plt.title('Exact Energies')
    #plt.plot(ts, energies_exact)
    #fig, ax = plt.subplots()
    #plt.title('Error in exact energy')
    #ax.plot(ts, np.abs(energies_exact - gs_energy))
    #ax.set_yscale('log')
    #plt.figure()
    #plt.title('Trace distances')
    #plt.plot(ts[1:], trace_dists)
    plt.show()


