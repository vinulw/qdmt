import sys
sys.path.insert(0, '../qdmt/')

import numpy as np
import os
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json

from loschmidt import loschmidt_paper


###############################################################################
# Load data for analysis
###############################################################################
print('Loading files...')
dataDir = Path('./data/07122023-111352')
dataFiles = os.listdir(dataDir)

print(dataFiles)
pattern = re.compile(r'A_t\d-\d{2}.npy')
dataFiles = [f for f in dataFiles if pattern.search(f)]

saveFig = True

configFile = Path(dataDir) / 'config.json'
with open(configFile, 'r') as cFile:
    config = json.load(cFile)

ts = []
As = []

pattern = re.compile(r'\d-\d{2}')
for f in tqdm(dataFiles):
    tqdm.write(f'\tLoading: {f}')
    filePath = os.path.join(dataDir, f)

    A = np.load(filePath, allow_pickle=True)
    As.append(A)

    t = float(pattern.search(f).group(0).replace('-', '.'))
    ts.append(t)

ts, As = zip(*sorted(zip(ts, As), key=lambda x: x[0]))

###############################################################################
# Analysis code including
# - Loschmidt echo plot (state + density matrix)
# - Error (trace distance with update density matrix)
###############################################################################
from analyse import exact_overlap, TrAB

A0 = np.copy(As[0])
stateLosch = []
densityLosch = []
N = 4 # N sites for TrAB

print('Calculating Losch...')
for A in tqdm(As):
    sLs = -1*np.log(exact_overlap(A0, A)**2)
    dLs = -1*np.log(np.real(TrAB(A0, A, N)))

    stateLosch.append(sLs)
    densityLosch.append(dLs)

# Prepare analytic
maxTime = ts[-1] + ts[1] - ts[0]
analyticsTs = np.linspace(0, maxTime, 250)
analyticLoschmidt = [np.real(loschmidt_paper(
                    t, config['g1'], config['g2'])) for t in analyticsTs]

plt.plot(analyticsTs, analyticLoschmidt, '--', label='Analytic' )
plt.plot(ts, stateLosch, 'x', label='State Losch')
plt.xlabel('Time Step')
plt.ylabel('Loschmidt Echo')
# plt.plot(ts, densityLosch, label='Density Losch')
plt.legend()

if saveFig:
    figPath = dataDir / 'loschmidt.png'
    plt.savefig(figPath)

plt.show()

