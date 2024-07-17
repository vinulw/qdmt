'''
Based on `https://tenpy.readthedocs.io/en/latest/examples/purification.html`.
'''
import numpy as np
from datetime import datetime
import tenpy
from tenpy.linalg import np_conserved as npc
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.purification_mps import PurificationMPS
from tenpy.algorithms.purification import PurificationTEBD
import matplotlib.pyplot as plt


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    order = 2
    dt = 0.02
    J = 1
    g = 0.5
    L = 20
    bc = 'finite'
    beta_max = 3.0

    model_params = {
        'L': L,
        'J':  J,
        'g':  g,
    }

    print('Generating model...')
    model = TFIChain(model_params)
    modelMPO = model.calc_H_MPO()

    print('Generating purification MPS...')
    psi = PurificationMPS.from_infiniteT(model.lat.mps_sites(), bc=bc)

    print('Calculating initial energy...')

    def calculate_energy_density(psi):
        expX = psi.expectation_value('Sx')
        expZ = psi.expectation_value('Sz')

        eng_density = -J*np.sum(expX[:-1]*expX[1:]) - g*np.sum(expZ)
        eng_density = eng_density / L

        return eng_density

    def calculate_norm(psi):
        expI = psi.expectation_value('Id')
        return np.sum(expI) / L


    print('Initialising TEBD engine...')
    eng_opts = {
        'trunc_params': {
            'chi_max': 100,
            'svd_min': 1.e-12
        },
        'order': order,
        'dt': dt,
        'N_steps': 1
    }
    engine = PurificationTEBD(psi, model, eng_opts)

    # Initalise data arrays
    steps = int(beta_max // (dt*2)) # Each step is 2*\beta
    betas = np.zeros(steps)
    energies = np.zeros(steps)
    norms = np.zeros(steps)

    print('Running imaginary time evolution...')
    energies[0] = calculate_energy_density(psi)
    norms[0] = calculate_norm(psi)

    for i in range(1, steps):
        betas[i] = betas[i-1] + 2.*dt
        engine.run_imaginary(dt)
        energies[i] = calculate_energy_density(psi)
        norms[i] = calculate_norm(psi)

    print('Saving data...')
    now = datetime.now().strftime('%d%m%y_%H%M%S')
    savePath = f'data/{now}_tenpy_thermal.npy'
    np.save(savePath, [betas, energies])

    print('Plotting...')
    plt.plot(betas, energies, 'x-')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'Energy density')

    plt.figure()
    kTs = 1/(betas[1:])
    plt.plot(kTs, energies[1:], 'x-')
    plt.xlabel(r'$k_BT$')
    plt.ylabel(r'Energy density')

    plt.figure()
    plt.plot(kTs, norms[1:], 'x-')
    plt.xlabel(r'$k_BT$')
    plt.ylabel(r'Norm')
    plt.show()




