import numpy as np
from scipy.stats import unitary_group
from environment import generate_environment_unitary
from environment import generate_state_transferMatrix, generate_right_environment
from ncon import ncon

def partial_density_matrix(A, R, N=1, ignored_indices=[]):
    '''
    Construct a partial density matrix ρA out of state and environment tensor.

    Args
    ----
    ignored_indices: list of indices that are not traced over.
    A: state tensor array
    R: environment tensor array
    N: Number of state tensors to include

    Output
    ------
    array representing ρA  with shape (d1, d2, ..., di, d1, d2, ...,  di) where
    i is the number of ignored indices
    '''
    Aconj = A.conj()
    if 0 in ignored_indices:
        contr = ncon([A, Aconj], ((1, -1, -3), (1, -2, -4)))
        contrindices = [-1, -2, 1, 2]
    else:
        contr = ncon([A, Aconj], ((1, 2, -1), (1, 2, -2)))
        contrindices = [1, 2]

    for i in range(N-1):
        n = i+1
        minInd = min(contrindices)
        if minInd > 0:
            minInd = 0
        if n in ignored_indices:
            contr = ncon([contr, A, Aconj], (contrindices, (1, minInd-1, minInd-3), (2, minInd-2, minInd-4)))
            contrindices = contrindices[:-2] + [minInd-1, minInd-2, 1, 2]
        else:
            contr = ncon([contr, A, Aconj], (contrindices, (1, 3, minInd-1), (2, 3, minInd-2)))

    contr = ncon([contr, R], (contrindices, (1, 2)))
    return contr

if __name__=="__main__":
    np.random.seed(1)
    d= 2
    D = 2 # Increment this to D=4 to fully represent time evolution

    # Generate a random state tensor
    stateU = unitary_group.rvs(d*D)
    A = stateU.reshape(d, D, d, D)
    A = A.transpose(2, 3, 0, 1)
    zero_state = np.eye(d)[0, :]
    A = ncon([A, zero_state], ((-1, -2, 1, -3), (1,))) # (Dl, σ, Dr)

    # Generate a random state tensor
    stateU = unitary_group.rvs(d*D)
    B = stateU.reshape(d, D, d, D)
    B = B.transpose(2, 3, 0, 1)
    zero_state = np.eye(d)[0, :]
    B = ncon([B, zero_state], ((-1, -2, 1, -3), (1,))) # (Dl, σ, Dr)

    print(np.allclose(A, B))

    # Generate the environment
    transferMatrix = generate_state_transferMatrix(A)
    R = generate_right_environment(transferMatrix)
    R = R.reshape(D, D)
    Rnorm = ncon([R], ((1, 1)))
    R = R/Rnorm

    # Perform contraction to verify overlaps
    contr = ncon([A, A.conj(), R], ((1, 2, 3), (1, 2, 4), (3, 4)))
    print(contr)

    # Generate ρ
    #   - 3 repetitions of the state tensor with the middle one being left open

    A = A.transpose(1, 0, 2) # Change to an array with shape (Dl, σ, Dr)
    ρA = partial_density_matrix(A, R, N=3, ignored_indices=[1])
    print(ρA.shape)
    B = B.transpose(1, 0, 2) # Change to an array with shape (Dl, σ, Dr)
    ρB = partial_density_matrix(B, R, N=3, ignored_indices=[1])

    print("Contracting ρA with itself...")
    overlap_ρAρA = ncon([ρA, ρA], ((1, 2), (2, 1)))
    print(overlap_ρAρA)

    print("Contracting ρA with ρB...")
    overlap_ρAρB = ncon([ρA, ρB], ((1, 2), (2, 1)))
    print(overlap_ρAρB)

    # Generate ρ
    #   - 4 repetitions of the state tensor with tensors 1, 2 being left open

    A = A.transpose(1, 0, 2) # Change to an array with shape (Dl, σ, Dr)
    ρA = partial_density_matrix(A, R, N=3, ignored_indices=[1, 2])
    print(ρA.shape)
    B = B.transpose(1, 0, 2) # Change to an array with shape (Dl, σ, Dr)
    ρB = partial_density_matrix(B, R, N=3, ignored_indices=[1, 2])

    print("Contracting ρA with itself...")
    overlap_ρAρA = ncon([ρA, ρA], ((1, 2, 3, 4), (2, 1, 4, 3)))
    print(overlap_ρAρA)

    print("Contracting ρA with ρB...")
    overlap_ρAρB = ncon([ρA, ρB], ((1, 2, 3, 4), (2, 1, 4, 3)))
    print(overlap_ρAρB)





