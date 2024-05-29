import numpy as np
import quimb
import quimb.tensor as qtn
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from tqdm import tqdm
from datetime import datetime
import json

# Below does not currently work, the states are not normalised
# def randomTensorsHaarMPS(shape, N):
#     '''
#     Sample `N` tensors in left canonical form with `shape = (Dl, ρ, Dr)`.
#     '''
#     Dl, ρ, Dr = shape
#     D = max(Dl*ρ, Dr)
#     newStates = [unitary_group.rvs(D)[:Dl*ρ, :Dr].reshape(Dl, ρ, Dr) for _ in range(N)]
#     newStates[0] = newStates[0][0, :, :]
#     newStates[-1] = newStates[-1][:, :, 0]
#     return newStates
#
# def randomHaarMPS(L, D):
#     tensors = randomTensorsHaarMPS((D, 2, D), L)
#     return qtn.MatrixProductState(tensors, shape='lpr')

def analyse_data(prefix, Ns=[500, 1000, 5000]):
    from sample_thermal import plotKbtEnergy
    dataFile = f'{prefix}_expHs.npy'
    configFile = f'{prefix}_config.json'
    with open(configFile, 'r') as fp:
        config = json.load(fp)

    L, D, N = config['L'], config['D'], config['N']

    title = r'Estimating $\expval{{H_T}}(k_BT)$ using finite Ransom MPS, L={}, D={}, Ns={}'.format(L, D, N)

    print(Ns)
    expHs = np.load(dataFile)
    plotKbtEnergy(expHs=expHs, Ns=Ns, title=title)


if __name__=="__main__":
    # prefix = 'data/290524_122955'
    # analyse_data(prefix)
    # assert False

    L = 20
    D = 4
    periodic = True
    g = 0.5
    MPOIsing = qtn.tensor_builder.MPO_ham_ising(L, j=-1.0, bx=g, cyclic=periodic)
    Hname = 'Ising'
    save = True
    print('Generated Hamiltonian...')

    Nsamples = 10000

    expHs = np.zeros(Nsamples)
    print('Sampling expectations...')
    for i in tqdm(range(Nsamples)):
        ψ = qtn.MPS_rand_state(L=L, bond_dim=D)
        ψ.left_canonize()
        ψH = ψ.H
        # print(ψ.show())
        # print('State norm: ', ψH @ ψ)

        ψ.align_(MPOIsing, ψH)
        expHs[i] = ((ψH & MPOIsing & ψ) ^ ...)

    if save:
        now = datetime.now().strftime('%d%m%y_%H%M%S')
        dataFile = f'data/{now}_expHs.npy'
        np.save(dataFile, expHs)
        print(f'Saved expectations as: {dataFile}')
        configFile = f'data/{now}_config.json'
        config = {
            'L': L,
            'D': D,
            'periodic': periodic,
            'g': g,
            'Hamiltonian': Hname,
            'N': Nsamples
        }
        with open(configFile, 'w+') as f:
            json.dump(config, f)



