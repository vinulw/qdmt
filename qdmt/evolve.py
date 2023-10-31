'''
Helper functions to evolve uMPS states
'''
import numpy as np
from ncon import ncon

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

    print(ρ_con)
    print(U_cons)
    print(U_dag_cons)

    breakpoint()

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


