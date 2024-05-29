import numpy as np
import matplotlib.pyplot as plt
from qdmt.hamiltonian import exact_thermal_energy
from tqdm import tqdm

if __name__=="__main__":
    fname='dataTFIM.txt'
    data = np.loadtxt(fname, delimiter=',')
    print(data.shape)

    betas = data[:, 0]
    expHs = data[:, 1]

    # Load exact data
    N = 100
    Ts = np.linspace(1e-5, 10, N)
    Betas = 1/Ts
    ETherms = np.zeros(N)
    J = 1.0
    g = 0.5

    print('Calculating exact curve...')
    for i, T in tqdm(enumerate(Ts), total=N):
        ETherms[i] = exact_thermal_energy(J, g, T)

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{physics}')
    plt.rcParams['font.size'] = 16

    plt.figure(figsize=(12, 8))
    plt.plot(betas, expHs, 'x-', label='purificationMPS')
    # plt.plot(Betas, ETherms, label='exact')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\expval{H}_{th}$')
    plt.legend()

    maskZero = data[:, 0] != 0.0
    expHs = data[maskZero, 1]
    kbTs = 1/data[maskZero, 0]

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ln1 = ax2.plot(Ts, ETherms, '--', label='exact', color='green')
    ln2 = ax1.plot(kbTs, expHs, 'x-', label='purificationMPS')
    ax1.set_xlabel(r'$k_B T$')
    ax1.set_ylabel(r'$\expval{H}_{th}$ Purification')
    ax2.set_ylabel(r'$\expval{H}_{th}$ Exact')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    # Calculating scaling
    exactExpHs = [exact_thermal_energy(J, g, T) for T in kbTs]
    scalings =  expHs / exactExpHs
    annText = 'Mean Scaling:  {:.4f}\nStd Scaling: {:.4f}'.format(np.mean(scalings),
                                                                  np.std(scalings))
    print(annText)
    plt.figure(figsize=(12, 8))
    plt.plot(kbTs, scalings, 'x--')
    plt.ylabel(r'$\expval{H}_{th}$ Ratio Purification / Exact')
    plt.xlabel(r'$k_B T$')
    plt.annotate(annText, (0.75, 0.75), xycoords='axes fraction')
    plt.show()
