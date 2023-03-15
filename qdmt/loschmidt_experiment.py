import numpy as np
from ground_state import Hamiltonian
from gs_experiment import rho_theta
from scipy.linalg import expm
from dmt_overlap import trace_distance, trρAρB
from ncon import ncon
from scipy.optimize import minimize
from tqdm import tqdm
import matplotlib.pyplot as plt


if __name__ == "__main__":
    import os
    cwd = os.getcwd()

    gs_param_file = 'gs_1.5_15032023115155.npy'
    gs_fpath = os.path.join(cwd, gs_param_file)
    print(gs_fpath)
    # Load the gs params
    θ0 = np.load(gs_fpath)

    dt = 0.01
    max_t = 2.0
    ts = np.arange(0, max_t, dt)
    g2 = 0.2
    d = 2
    D = 2
    N = 4


    H = Hamiltonian({'ZZ':-1, 'X': g2}).to_matrix()

    U = expm(-1j*H*dt*2.0)
    U = U.reshape(2, 2, 2, 2)

    def objective_func_trace(θdt, ρ0dt):
        ρt = rho_theta(θdt, N, d, D)
        cost = np.abs(trace_distance(ρt, ρ0dt))
        return cost

    ρ0 = rho_theta(θ0, N, d, D)

    ρ_curr = np.copy(ρ0)
    θ_curr = np.copy(θ0)

    loschmidts = np.zeros(ts.shape[0], dtype=complex)
    loschmidts[0] = trρAρB(ρ0, ρ_curr)

    for i, t in tqdm(enumerate(ts[1:]), total=len(ts[1:])):
        ρcurrdt = ncon([U, U, ρ_curr, U.conj(), U.conj()],
                       ((-1, -3, 1, 3), (-5, -7, 5, 7),
                       (1, 2, 3, 4, 5, 6, 7, 8),
                       (-2, -4, 2, 4), (-6, -8, 6, 8)))
                       #(2, 4, -2, -4), (6, 8, -6, -8)))

        res = minimize(objective_func_trace, θ_curr, args=(ρcurrdt))

        θ_curr = res.x
        ρ_curr = rho_theta(θ_curr, N, d, D)
        loschmidts[i+1] = trρAρB(ρ0, ρ_curr)

        if res.success is False:
            tqdm.write(f"Step: {i}")
            tqdm.write(f"   Obj func : {res.fun}")
            tqdm.write(f"   Message: {res.message}")

    from loschmidt import loschmidt_paper
    ltimes = np.linspace(0, max_t, 200)
    correct_ls = [loschmidt_paper(t, 1.5, 0.2) for t in ltimes]
    loschmidts = np.abs(loschmidts)

    fig, ax = plt.subplots()
    ax.plot(ts, -1*np.log(loschmidts), '--', color='b', label='qDMT')
    ax.set_ylabel('qDMT', color='b')

    ax2 = ax.twinx()
    ax2.plot(ltimes, correct_ls, '-.', color='g', label='Analytic')
    ax2.set_ylabel('Analytic', color='g')
    plt.show()








