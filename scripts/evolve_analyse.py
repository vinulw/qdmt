import sys
sys.path.insert(0, '../qdmt/')

import numpy as np
import os
import re
from tqdm import tqdm
import matplotlib.pyplot as plt


###############################################################################
# Load data for analysis
###############################################################################
print('Loading files...')
dataDir = './data/14112023-114548/'
dataFiles = os.listdir(dataDir)

pattern = re.compile(r'A_t\d-\d{2}.npy')
dataFiles = [f for f in dataFiles if pattern.search(f)]

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
    sLs = exact_overlap(A0, A)
    dLs = np.real(TrAB(A0, A, N))

    stateLosch.append(sLs)
    densityLosch.append(dLs)

plt.plot(ts, stateLosch, label='State Losch')
plt.plot(ts, densityLosch, label='Density Losch')
plt.legend()
plt.show()

