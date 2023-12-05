import sys
sys.path.insert(0, '../qdmt/')

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys

# Add qdmt
from analyse import exact_overlap2
from loschmidt import loschmidt_paper


if __name__ == "__main__":

    prefix = None
    if len(sys.argv) > 1:
        prefix = sys.argv[1]

    assert prefix is not None, 'Please provide a prefix for the run...'

    saveDir = Path('./data/tenpy_timeev/')

    print(f'Prefix = {prefix}')
    # Load config
    with open(saveDir / f'{prefix}_config.json', 'r') as f:
        config = json.load(f)

    maxTime = config['maxTime']
    dt = config['dt']
    ts = np.arange(0, maxTime, dt)

    g1 = config['g1']
    g2 = config['g2']

    # Load tensors
    dataPath = saveDir / f'{prefix}-Ats.npy'
    Ats = np.load(dataPath, allow_pickle=True)
    A0 = Ats[0]

    bondDims = [A[0].shape[0] for A in Ats]
    print(f'Maximum bond dimension: {max(bondDims)}')

    # Analyse the results
    loschmidt = []
    i = 0
    for A in tqdm(Ats):
        ls = exact_overlap2(A0, A)
        ls = -1*np.log(ls)
        loschmidt.append(ls)
        i += 1

    analyticsts = np.linspace(0, maxTime, 250)
    analyticLoschmidt = [np.real(loschmidt_paper(t, g1, g2)) for t in analyticsts]

    plt.plot(analyticsts, analyticLoschmidt, '--', label='analytic')
    plt.plot(ts, loschmidt, 'x', label='iTEBD')
    plt.legend()
    plt.show()
