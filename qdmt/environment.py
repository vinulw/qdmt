import numpy as np
from ncon import ncon
from scipy.stats import unitary_group
from scipy.linalg import eig, null_space


def generate_transferMatrix(A, W):
    """
    For a given state tensor and evolution, calculate the transfer matrix:

    --- A ---
        |
    --- W ---
        |
    --- W*---
        |
    --- A*---

    Args:
        A (array) : State tensor with dimension (σA, DAl, DAr)
        W (array) : Time evolution tensor (σW, lW, DWl, DWr)
    Returns:
        Transfer matrix (array) : Transfer matrix with dimension (DAl**2
                * DWl**2, DAr**2 * DWr**2)
    """

    σA, DAl, DAr = A.shape
    σW, lW, DWl, DWr = W.shape

    TM = ncon([A, W, W.conj(), A.conj()],
            ((1, -1, -5), (2, 1, -2, -6), (2, 3, -3, -7), (3, -4, -8)))

#    print("Verifying unitarity")
#    TMI = ncon([TM,], ((1, 2, 2, 1, -2, -1, -3, -4),)) # Need swap on one of the sides to get it to be identity
#    TMI = TMI.reshape(DAr*DWr, DAr*DWr)
#    I = np.eye(DAr*DWr)
#    print(TMI)
#    print(np.allclose(I, TMI))

    TM = TM.reshape(DAl**2*DWl**2, DAr**2*DWr**2)

    return TM

if __name__=="__main__":
    d = 2
    D = 2
    U = unitary_group.rvs(d*D)
    U = U.reshape(d, D, d, D)
    U = U.transpose(2, 3, 0, 1)
    zero_state = np.eye(d)[0, :]

    A = ncon([U, zero_state], ((-1, -2, 1, -3), (1,)))

    # Verifying the left canonicalisation condition
    AI = ncon([A, A.conj()], ((1, 2, -1), (1, 2, -2)))
    print("Verifying state left canonicalisation")
    print('    ', np.allclose(AI, np.eye(D)))

    # Generating time evolution unitary
    W = unitary_group.rvs(d*D)
    W = W.reshape(d, D, d, D)
    W = W.transpose(0, 2, 1, 3)

    # Verifying the left canonicalisation of W
    WI = ncon([W, W.conj()], ((1, -1, 2, -2), (1, -3, 2, -4)))
    WI = WI.reshape(d*D, d*D)
    print("Verifying operator unitary condition")
    print('    ', np.allclose(WI, np.eye(d*D)))

    # Generating transfer matrix
    transferMatrix = generate_transferMatrix(A, W)

    # Find the right environment
    evals, evecs = eig(transferMatrix)
    magEvals = np.abs(evals)
    maxArg = np.argmax(magEvals)

    print(f"Max arg: {maxArg}")
    R = evecs[:, maxArg]

    print("Maximum eigenvector found...")
    print("Checking if it satisfies T R = λ R")

    TR = transferMatrix @ R
    λarr = TR / R
    print('    ', np.isclose(np.min(λarr), np.max(λarr))) # Checking maxima and minima of λ are close

    # Verifying that R is hermitian
    R = R.reshape(D, D, D, D) # DAr, DWr, DWconjr, DAconjr
    R = R.transpose(0, 1, 3, 2)
    R = R.reshape(D*D, D*D)

    Rdagger = R.conj().T

    print("Verifying R is Hermitian...")
    print('    ', np.allclose(R, Rdagger))

    U, S, V = np.linalg.svd(R)
    sqrtS = np.sqrt(S)

    S = np.diag(S)
    sqrtS = np.diag(sqrtS)

    print(np.allclose(S, sqrtS @ sqrtS))

    U = U @ sqrtS

    U = U.reshape(-1, )

#    V = np.zeros([U.shape[0], U.shape[0]], dtype=complex)
#    V[:, 0] = U
#    V, _ = np.linalg.qr(V)

    s = np.eye(U.shape[0])[0]
    def create_unitary(v):
        dim = v.size
        e1 = np.zeros(dim, dtype=complex)
        e1[0] = 1
        w = v/np.linalg.norm(v) - e1
        return np.eye(dim, dtype=complex) - 2*((np.dot(w.T, w))/(np.dot(w, w.T)))

    # See this page for some suggestions on making a unitary out of a vector
    # https://math.stackexchange.com/questions/4160055/create-a-unitary-matrix-out-of-a-column-vector
    def create_Householder_matrix(v):
        v = v / np.linalg.norm(v)
        dim = v.shape[0]
        return np.eye(dim) - 2*np.outer(v, v.conj().T)

    V = create_Householder_matrix(U)

    UV = V @ s
    print('U : UV : U / UV')
    for i in range(U.shape[0]):
        print(f'{U[i]}  :  {UV[i]}  :  {(U/UV)[i]}')


    # Checking V reproduced U state
    print("Checking V reproduces U state")
    print('    ', np.allclose(U, UV))

    # Verifying that V is unitary
    Vdagger = V.conj().T
    print('Verifying V is unitary')
    print('    ', np.allclose(np.eye(V.shape[0]), V @ Vdagger))
    print('    ', np.allclose(np.eye(V.shape[0]), Vdagger @ V))





