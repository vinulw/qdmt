import numpy as np
from scipy.stats import unitary_group
from ncon import ncon
from qdmt.hamiltonian import TransverseIsing
from qdmt.uMPSHelpers import rightFixedPoint
from tqdm import tqdm
from datetime import datetime

import matplotlib.pyplot as plt

def sampleUnitaryTensors(shape, N):
    '''
    Sample `N` tensors in left canonical form with `shape = (Dl, ρ, Dr)`.
    '''
    Dl, ρ, Dr = shape
    D = max(Dl*ρ, Dr)
    newStates = [unitary_group.rvs(D)[:Dl*ρ, :Dr].reshape(Dl, ρ, Dr) for _ in tqdm(range(N))]
    return newStates

def isLeftCanonical(A):
    Dl, ρ, Dr = A.shape
    AA = ncon([A, A.conj()], ((1, 2, -1), (1, 2, -2)))

    I = np.eye(Dr)
    return np.allclose(I, AA)

def test_sampleUnitaryTensors():
    N = 10
    shape = (4, 2, 2)

    print('Sampling As...')
    As = sampleUnitaryTensors(shape, N)

    print('Verifying left canonical...')
    for A in As:
        assert isLeftCanonical(A)

    print('Is left canonical')

def expectationLeftCanonical(A, H):
    '''
    Calculate the expectation of `H` for a left canonical uMPS represented by
    state tensor `A`.
    '''
    Dl, ρ, Dr = A.shape
    r = rightFixedPointNormalised(A)

    nSites = H.shape[0] // ρ
    Hten = H.reshape(*[ρ] * (2*nSites))

    tensors = [A]*nSites + [Hten] + [A.conj()]*nSites + [r]

    Acontr = [[i*4, i*4+2, (i+1)*4] for i in range(nSites)]
    ADagcontr = [[i*4+1, i*4+3, (i+1)*4+1] for i in range(nSites)]
    Acontr[0][0] = ADagcontr[0][0]
    Hcontr = [a[1] for a in Acontr] + [a[1] for a in ADagcontr]
    rContr = [Acontr[-1][-1], ADagcontr[-1][-1]]
    contr = Acontr + [Hcontr] + ADagcontr + [rContr]

    return ncon(tensors, contr)

def rightFixedPointNormalised(A):
    '''
    Calculate right fixed point and ensure it is normalised for left canonical
    matrices.
    '''
    r = rightFixedPoint(A)
    return r / np.trace(r)


def test_rightFixedPointNormalised():
    shape = (4, 2, 4)
    N = 10
    As = sampleUnitaryTensors(shape, N)

    Rs = [rightFixedPointNormalised(A) for A in As]
    selfOverlaps = [ncon([r,], ((1, 1),)) for r in Rs]

    for ov in selfOverlaps:
        assert np.allclose(ov, 1+0j)


if __name__=="__main__":
    # # Sampling states
    shape = (4, 2, 4)
    N = 10
    As = sampleUnitaryTensors(shape, N)

    # # Generating Hamiltonians
    # print('Generating TFIM Hamiltonian')
    # J = 1.0
    # g = 0.2
    # n = 2
    # H = TransverseIsing(J, g, n)
    # print(H.shape)

    # # Verifying contraction for expectation value
    # nSites = 3
    # print(f'nSites: {nSites}')
    # Acontr = [[i*4, i*4+2, (i+1)*4] for i in range(nSites)]
    # ADagcontr = [[i*4+1, i*4+3, (i+1)*4+1] for i in range(nSites)]
    # Acontr[0][0] = ADagcontr[0][0]

    # Hcontr = [a[1] for a in Acontr] + [a[1] for a in ADagcontr]
    # rContr = [Acontr[-1][-1], ADagcontr[-1][-1]]
    # print(f'Acontr: {Acontr}')
    # print(f'ADagcontr: {ADagcontr}')
    # print(f'Hcontr: {Hcontr}')
    # print(f'rContr: {rContr}')

    # # Verifying right fixed point is normalised
    # Rs = [rightFixedPointNormalised(A) for A in As]
    # selfOverlaps = [ncon([r,], ((1, 1),)) for r in Rs]
    # print(selfOverlaps)

    # test_rightFixedPointNormalised()

    # Calculating expectation values and save.
    # save = True
    # now = datetime.now().strftime('%d%m%y_%H%M%S')

    # shape = (4, 2, 4)
    # N = 100000
    # print('Collecting thermal distributions...')
    # print('Sampling As...')
    # As = sampleUnitaryTensors(shape, N)

    # # print('Preparing H...')
    # H = TransverseIsing(J=1.0, g=0.2, n=2)
    # print('Calculating expectations...')
    # expHs = np.array([expectationLeftCanonical(A, H) for A in tqdm(As)])

    # if save:
    #     savePath = f'data/{now}_As.npy'
    #     print(f'Saving As as: {savePath}')
    #     np.save(savePath, As)
    #     savePath = f'data/{now}_expHs.npy'
    #     print(f'Saving expHs as: {savePath}')
    #     np.save(savePath, expHs)

    # breakpoint()

    # Analyse thermal expectation
    prefix = 'data/120324_163721_' # 100 000 samples
    print(f'Analysing data from run: {prefix}')
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{physics}')
    expHs = np.load(prefix + 'expHs.npy')

    Ns = [1000, 5000, 10000, 50000]
    repeats = 10

    for N in Ns:
        if len(expHs) < N:    # skip N if bigger than no samples
            continue
        nBetas = 50
        betas = np.logspace(-1, 2, nBetas)
        expHTherm = np.zeros((nBetas, repeats), dtype=complex)
        for j in range(repeats):
            expHs_ = np.random.choice(expHs, size=N, replace=False)

            # Sampling thermal distributions
            for i, beta in enumerate(betas):
                expBHs = np.exp(-beta*expHs_)
                expHTherm[i][j] = np.sum(np.dot(expBHs, expHs_)) / np.sum(expBHs)

        means = np.mean(np.real(expHTherm), axis=1)
        stds = np.std(np.real(expHTherm), axis=1)

        # print(f'Expectations real: ', np.allclose(np.imag(expHs), 0))
        plt.errorbar(betas, means, yerr=stds,
                      fmt='x-', label=f'N: {N}', capsize=3.0)

    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\expval{H_{T}}$')
    plt.title(r'Estimating $\expval{H_T}(\beta)$ using boosting. $N_{samp} = 10^5$')
    plt.legend()
    plt.show()
