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
        if n in ignored_indices:
            contr = ncon([contr, A, Aconj], (contrindices, (1, minInd-1, minInd-3), (2, minInd-2, minInd-4)))
            contrindices = contrindices[:-2].extend([minInd-1, minInd-2, 1, 2])
        else:
            contr = ncon([contr, A, Aconj], (contrindices, (1, 3, minInd-1), (2, 3, minInd-2)))

    contr = ncon([contr, R], (contrindices, (1, 2)))
    return contr

if __name__=="__main__":
    d= 2
    D = 2 # Increment this to D=4 to fully represent time evolution

    # Generate a random state tensor
    stateU = unitary_group.rvs(d*D)
    A = stateU.reshape(d, D, d, D)
    A = A.transpose(2, 3, 0, 1)
    zero_state = np.eye(d)[0, :]
    A = ncon([A, zero_state], ((-1, -2, 1, -3), (1,))) # (Dl, σ, Dr)

    # Generate the environment
    transferMatrix = generate_state_transferMatrix(A)
    R = generate_right_environment(transferMatrix)
    R = R.reshape(D, D)

    # Perform contraction to verify overlaps
    contr = ncon([A, A.conj(), R], ((1, 2, 3), (1, 2, 4), (3, 4)))
    print(contr)



