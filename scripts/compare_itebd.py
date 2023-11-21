import sys
sys.path.insert(0, '../qdmt/')

import numpy as np
import os
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

from pathlib import Path

from analyse import exact_overlap

###############################################################################
# Compare the output from iTEBD with the DMT algorithm
###############################################################################

print('Loading data...')
dmtDataDir = Path('./data/14112023-114548/')
iTEBDdata = Path('./data/tenpy_timeev/20112023-151759-Ats.npy')

# Load the DMT tensors
AsDmt = []
tsDmt = []
pattern = r'A_t(\d-\d{2}).npy'
for filePath in dmtDataDir.rglob('*.npy'):
    matched = re.search(pattern, str(filePath))
    if matched:
        tStep = float(matched.group(1).replace('-', '.'))
        AsDmt.append(np.load(filePath, allow_pickle=True))
        tsDmt.append(tStep)

# Sort the DMT tensors
tsDmt, AsDmt = zip(*sorted(zip(tsDmt, AsDmt), key=lambda x: x[0]))

# Load the itebd tensors
AsTenpy = np.load(iTEBDdata, allow_pickle=True)

print('Comparing overlaps')
overlaps = []
for i in range(min((len(AsDmt), len(AsTenpy)))):
    print(f'Time Step: {tsDmt[i]}')
    overlaps.append(exact_overlap(AsDmt[i], AsTenpy[i]))
    print(f'\tOverlap: {overlaps[-1]}')

plt.plot(tsDmt, overlaps, '.')
plt.xlabel('Time')
plt.ylabel('Overlap with iTEBD')
plt.show()


