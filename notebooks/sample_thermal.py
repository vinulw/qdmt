import numpy as np
from scipy.stats import unitary_group
from ncon import ncon
from qdmt.hamiltonian import TransverseIsing
from qdmt.hamiltonian import exact_thermal_energy
from qdmt.uMPSHelpers import rightFixedPoint
from tqdm import tqdm
from datetime import datetime
from functools import reduce

from numpy import kron, eye

import matplotlib.pyplot as plt

# define Paulis
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = 1j*np.array([[0, -1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = eye(2, dtype=complex)

norm = np.sqrt(2)

S = {'I': I/norm, 'X': X/norm, 'Y': Y/norm, 'Z': Z/norm}


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

def generateHamiltonianPeriodic(J, g, n):
    '''
    Generate n qubit TFIM Hamiltonian.
    '''

    h = np.zeros((2**n, 2**n)) + 0j

    for i in range(n-1):
        pString = ['I'] * n
        pString[i] = 'Z'
        pString[i+1] = 'Z'
        hzz = reduce(kron, [S[j] for j in pString])

        pString = ['I'] * n
        pString[i] = 'X'
        hxx = reduce(kron, [S[j] for j in pString])

        h += -J * hzz - g * hxx

    pString = ['I'] * n
    pString[n-1] = 'Z'
    pString[0] = 'Z'
    hzz = reduce(kron, [S[j] for j in pString])

    pString = ['I'] * n
    pString[n-1] = 'X'
    hxx += reduce(kron, [S[j] for j in pString])
    h += -J * hzz - g*hxx
    return h

def plotKbtEnergy(expHs):
    # Analyse thermal expectation
    # expHs = expHs / 2 # Due to defintion of Pauli spin operators, change the Ham

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{physics}')
    plt.rcParams['font.size'] = 18

    Ns = [1000, 5000, 10000, 50000]
    # Ns = [1000]
    repeats = 10

    print('Sampling thermal dist...')
    for N in Ns:
        if len(expHs) < N:    # skip N if bigger than no samples
            continue
        nBetas = 50
        kTs = np.logspace(-3, 0, nBetas)
        expHTherm = np.zeros((nBetas, repeats), dtype=complex)
        for j in range(repeats):
            expHs_ = np.random.choice(expHs, size=N, replace=False)

            # Sampling thermal distributions
            for i, kT in enumerate(kTs):
                expBHs = np.exp(-expHs_/kT)
                expHTherm[i][j] = np.sum(np.dot(expBHs, expHs_)) / np.sum(expBHs)

        # expHTherm = expHTherm / 2
        means = np.mean(np.real(expHTherm), axis=1)
        stds = np.std(np.real(expHTherm), axis=1)

        # print(f'Expectations real: ', np.allclose(np.imag(expHs), 0))
        plt.errorbar(kTs, means, yerr=stds,
                      fmt='x-', label=f'N: {N}', capsize=3.0)

    # Plot exact results
    J = 1
    g = 0.5
    N = 100

    Ts = np.linspace(1e-5, 10, N)
    ETherms = np.zeros(N)

    print('Calculating exact curve...')
    for i, T in tqdm(enumerate(Ts), total=N):
        ETherms[i] = exact_thermal_energy(J, g, T)

    # plt.plot(Ts, ETherms, '-', label='exact')
    rescaledTs = Ts / 10
    plt.plot(rescaledTs, ETherms, '-', label='exact')
    plt.xlabel(r'$k_BT$')
    plt.ylabel(r'$\expval{H_{T}}$')
    plt.title(r'Estimating $\expval{H_T}(k_BT)$ using boosting. $N_{samp} = 10^5$')
    plt.legend()
    plt.show()

def plotEnergyHist(expHs, dataFile=None):
    from scipy.stats import norm

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{physics}')

    plt.figure(figsize=(12, 8))
    plt.rcParams['font.size'] = 16

    expHs = np.real(expHs)

    mu, std = norm.fit(expHs)
    # print(mu, std)
    # print(norm.ppf(0.99, mu, std))
    # print(norm.ppf(0.01, mu, std))

    n, bins, patches = plt.hist(expHs, bins=100, density=True)

    normFit = norm.pdf(bins, mu, std)
    plt.plot(bins, normFit, '--')
    plt.title(r'Histogram of $\expval{{H}}$: Haar Random Unitaries, ${:.2E}$ samples'.format(len(expHs)))
    plt.xlabel(r'$\expval{H}$')
    plt.ylabel('Probability density')

    plt.annotate(r'Fit: $\mathcal{{N}}(\mu = {0:.4f}, \sigma = {1:.4f})$'.format(mu, std),
                 xy=(std*1.1, max(normFit)))

    if dataFile is not None:
        plt.annotate(f'Data File: {dataFile}', xy=(0.05, 0.025), xycoords='figure fraction')

    plt.show()



if __name__=="__main__":
    save = True
    now = datetime.now().strftime('%d%m%y_%H%M%S')


    prefix = None # 'data/020424_153954_' # 100 000 samples
    dataFile = None
#     prefix = 'data/010524_123814_'
    if prefix is None:
        J = 1.0
        g = 0.5
        D = 8

        shape = (D, 2, D)
        N = int(10**5)
        # N = 10000
        print('Collecting thermal distributions...')
        print('Sampling As...')
        As = sampleUnitaryTensors(shape, N)

        # print('Preparing H...')
        # H = TransverseIsing(J=J, g=g, n=2)
        H = generateHamiltonianPeriodic(J=J, g=g, n=2)
        print('Calculating expectations...')
        expHs = np.array([expectationLeftCanonical(A, H) for A in tqdm(As)])

        if save:
            savePath = f'data/{now}_As.npy'
            print(f'Saving As as: {savePath}')
            np.save(savePath, As)
            dataFile = f'data/{now}_expHs.npy'
            print(f'Saving expHs as: {dataFile}')
            np.save(dataFile, expHs)
    else:
        print(f'Analysing data from run: {prefix}')
        dataFile = prefix + 'expHs.npy'
        expHs = np.load(dataFile)

    plotEnergyHist(expHs, dataFile)

