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

def test_partial_density_matrix():
    '''
    Checking if Tr ρA == 1
    '''
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
    Rnorm = ncon([R], ((1, 1)))
    R = R/Rnorm

    A = A.transpose(1, 0, 2) # Change to an array with shape (Dl, σ, Dr)
    ρA = partial_density_matrix(A, R, N=3, ignored_indices=[1, 2])

    overlap_ρA = ncon([ρA], ((1, 1, 2, 2)))

    assert np.allclose(overlap_ρA, 1.0+0j)

def trace_distance(ρA, ρB):
    '''
    Calculate trace distance between ρA and ρB i.e. Tr((ρA - ρB)^2)

    Args
    ----
    ρA, ρB : Arrays representing density matrices A and B respectively. Note
            the shape of ρ should be (d1, d1, d2, d2, d3, d3, ...) where di represents
            the dimension of index i of the subspace of the density matrix. Each
            subspace appears in a pair representing the input and output legs of the
            density matrix. I.e. the density matrix is organised as in the diagram below:

            d1 --- ⎸     ⎸ --- d1
                   ⎸     ⎸
            d2 --- ⎸     ⎸ --- d2
                   ⎸  ρ  ⎸
                :
            dn --- ⎸     ⎸ --- dn
    '''
    assert ρA.shape == ρB.shape , 'ρA and ρB need to have matching dimensions'

    n = len(ρA.shape)

    assert n % 2 == 0

    contr0 = list(range(1, n+1))
    contr1 = []
    for i in range(n):
        if i % 2 == 0:
            contr1.append(contr0[i+1] )
        else:
            contr1.append(contr0[i-1])

    trρAρA = ncon([ρA, ρA], (contr0, contr1))
    trρAρB = ncon([ρA, ρB], (contr0, contr1))
    trρBρB = ncon([ρB, ρB], (contr0, contr1))

    return trρAρA + trρBρB - 2*trρAρB

def test_trace_distance():
    np.random.seed(1)
    d= 2
    D = 2 # Increment this to D=4 to fully represent time evolution

    # Generate a random state tensor
    stateU = unitary_group.rvs(d*D)
    A = stateU.reshape(d, D, d, D)
    A = A.transpose(2, 3, 0, 1)
    zero_state = np.eye(d)[0, :]
    A = ncon([A, zero_state], ((-1, -2, 1, -3), (1,))) # (Dl, σ, Dr)

    # Copy state
    B = np.copy(A)

    # Generate the environment
    transferMatrix = generate_state_transferMatrix(A)
    R = generate_right_environment(transferMatrix)
    R = R.reshape(D, D)
    Rnorm = ncon([R], ((1, 1)))
    R = R/Rnorm

    # Copy environment
    RB = np.copy(R)


    A = A.transpose(1, 0, 2) # Change to an array with shape (Dl, σ, Dr)
    ρA = partial_density_matrix(A, R, N=3, ignored_indices=[1, 2])
    B = B.transpose(1, 0, 2) # Change to an array with shape (Dl, σ, Dr)
    ρB = partial_density_matrix(B, RB, N=3, ignored_indices=[1, 2])

    assert np.allclose(trace_distance(ρA, ρB), 0+0j)

    stateU = unitary_group.rvs(d*D)
    B = stateU.reshape(d, D, d, D)
    B = B.transpose(2, 3, 0, 1)
    zero_state = np.eye(d)[0, :]
    B = ncon([B, zero_state], ((-1, -2, 1, -3), (1,))) # (Dl, σ, Dr)

    transferMatrix = generate_state_transferMatrix(B)
    RB = generate_right_environment(transferMatrix)
    RB = R.reshape(D, D)
    Rnorm = ncon([RB], ((1, 1)))
    RB = RB/Rnorm

    B = B.transpose(1, 0, 2) # Change to an array with shape (Dl, σ, Dr)
    ρB = partial_density_matrix(B, RB, N=3, ignored_indices=[1, 2])

    dist = trace_distance(ρA, ρB)

    assert np.real(dist) > 0


if __name__=="__main__":
    print("Testing partial density matrix")
    test_partial_density_matrix()
    print("   Passed.")

    print("Testing trace distance...")
    test_trace_distance()
    print("   Passed.")
