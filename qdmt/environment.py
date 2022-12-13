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

def generate_right_environment(transferMatrix):
    """
    For a given transfer matrix produce the vector that gives the right
    environment. This corresponds to the right eigenvector with the highest
    eigenvalue.
    """
    evals, evecs = eig(transferMatrix)
    magEvals = np.abs(evals)
    maxArg = np.argmax(magEvals)

    R = evecs[:, maxArg]
    return R

def embed_state_in_unitary(ψ):
    """
    Embed a state ψ into a unitary V in the |0> state such that V|0> = ψ.

    Note this requires the <0|ψ> term in the vector to be real.

    TODO: Try to implement a Gram Schmidt version of this algorithm.
    """
    dim = ψ.shape[0]
    zero = np.eye(dim)[0]
    assert np.isclose(np.imag(np.dot(zero, ψ)), 0.0), "First element of ψ needs to be real"
    v = (zero + ψ) / np.sqrt(2*(1 + np.dot(zero, ψ)))
    return 2*np.outer(v, v.conj()) - np.eye(dim)

def generate_environment_unitary(A, W, D):
    """
    For a given state tensor and evolution, produce the environment unitary V.

    This only works when the bond dimension is a constant D for A and W
    """
    transferMatrix = generate_transferMatrix(A, W)

    R = generate_right_environment(transferMatrix)

    # Need to rearrange legs to make R Hermitian
    R = R.reshape(D, D, D, D) # DAr, DWr, DWconjr, DAconjr
    R = R.transpose(0, 1, 3, 2)
    R = R.reshape(D*D, D*D)

    # Produce the state U to embed in V
    U, S, V = np.linalg.svd(R)
    sqrtS = np.sqrt(S)

    S = np.diag(S)
    sqrtS = np.diag(sqrtS)

    U = U @ sqrtS

    U = U.reshape(-1, )
    # State needs to be normalised to be embedded in V
    U = U / np.linalg.norm(U)

    V = embed_state_in_unitary(U)

    return V

def generate_random_state(d, D):
    """
    Generate a random state tensor A (σ, Dl, Dr) with the proper canoncalisation.
    """
    U = unitary_group.rvs(d*D)
    U = U.reshape(d, D, d, D)
    U = U.transpose(2, 3, 0, 1)
    zero_state = np.eye(d)[0, :]

    A = ncon([U, zero_state], ((-1, -2, 1, -3), (1,)))
    return A

def generate_ranom_operator(d, D):
    """
    Generate a random operator tensor W (σ, Dl, l, Dr) = (d, D, d, D) with the
    proper canonicalisation.
    """
    W = unitary_group.rvs(d*D)
    W = W.reshape(d, D, d, D)
    W = W.transpose(0, 2, 1, 3)
    return W

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

    from copy import copy
    R_ = copy(R)
    R = R.reshape(D, D, D, D) # DAr, DWr, DWconjr, DAconjr
    R = R.transpose(0, 1, 3, 2)
    R = R.reshape(D*D, D*D)

    print("Inverting R operations...")

    Rinv = R.reshape(D, D, D, D)
    Rinv = Rinv.transpose(0, 1, 3, 2)
    Rinv = Rinv.reshape(-1, )

    print(np.allclose(Rinv, R_))

    Rdagger = R.conj().T

    # Verifying that R is hermitian
    print("Verifying R is Hermitian...")
    print('    ', np.allclose(R, Rdagger))

    U, S, V = np.linalg.svd(R)
    sqrtS = np.sqrt(S)

    S = np.diag(S)
    sqrtS = np.diag(sqrtS)

    U = U @ sqrtS

    RU = U @ U.conj().T

    print("U found reproduces R:   ")
    print("   ", np.allclose(R, RU))

    U = U.reshape(-1, )
    U = U / np.linalg.norm(U)

#    V = np.zeros([U.shape[0], U.shape[0]], dtype=complex)
#    V[:, 0] = U
#    V, _ = np.linalg.qr(V)

    s = np.eye(U.shape[0])[0]

    def unitary_from_state(ψ, debug=False):
        '''
        Try and generate a unitary with the U|0> = |ψ>

        Inspired by : https://quantumcomputing.stackexchange.com/questions/10239/how-can-i-fill-a-unitary-knowing-only-its-first-column

        TODO:
            - Alternative method could be done using Gram Schmidt process
        '''
        dim = ψ.shape[0]
        zero = np.eye(dim)[0]
        if debug:
            print("Zeroth element of ψ:")
            print(np.dot(zero, ψ))
        assert np.isclose(np.imag(np.dot(zero, ψ)), 0.0), "First element of ψ needs to be real"
        v = (zero + ψ) / np.sqrt(2*(1 + np.dot(zero, ψ)))
        return 2*np.outer(v, v.conj()) - np.eye(dim)

    V = unitary_from_state(U)

    UV = V @ s

#    print('U : UV : U / UV')
#    for i in range(U.shape[0]):
#        print(f'{U[i]}  :  {UV[i]}  :  {(U/UV)[i]}')

    # Checking V reproduced U state
    print("Checking V reproduces U state")
    print('    ', np.allclose(U, UV))

    # Verifying that V is unitary
    Vdagger = V.conj().T
    print('Verifying V is unitary')
    print('    ', np.allclose(np.eye(V.shape[0]), V @ Vdagger))
    print('    ', np.allclose(np.eye(V.shape[0]), Vdagger @ V))

    # Verify that the reporduced U is a eigenvector for the transfer matrix
    print("Verifying U from V produces environment...")
    UV = UV.reshape(D*D, -1)
    envU = ncon([UV, UV.conj()], ((-1, 1), (-2, 1)))
    envU = envU.reshape(D, D, D, D)
    envU = envU.transpose(0, 1, 3, 2)
    envU = envU.reshape(-1)

    λenv = (transferMatrix @ envU) / envU
    maxλ = np.max(λenv)
    minλ = np.min(λenv)

#    print("R results:  ")
#    print(np.linalg.norm(R))
#    print(np.max(λR))
#    print(np.min(λR))
#    print("Env results:  ")
#    print(np.linalg.norm(λenv))
#    print(np.max(λenv))
#    print(np.min(λenv))

    print('   ', np.isclose(maxλ, minλ))
