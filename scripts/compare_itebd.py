import sys
sys.path.insert(0, '../qdmt/')

import numpy as np
import os
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from ncon import ncon

from pathlib import Path

from analyse import exact_overlap

###############################################################################
# Compare the output from iTEBD with the DMT algorithm
# - Calculate the trace distance between the two $A$s over a given patch size
#   $N$.
###############################################################################

print('Loading data...')
dmtDataDir = Path('./data/30052024-174627/')
iTEBDdata = Path('./data/tenpy_timeev/05122023-175825-Ats.npy')

saveFig = True

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
# Note that the patch size used during the itebd code is 2.
# So a two sites are combined to give a single site tensor.
# Patch sizes are actually size $2*N$.
AsTenpy = np.load(iTEBDdata, allow_pickle=True)

# Combine adjacent sites
As2Tenpy = []
for As in AsTenpy:
    Dl, p, _ = As[0].shape
    _, _, Dr = As[1].shape

    contr = ((-1, -2, 1), (1, -3, -4))
    A2Site = ncon([As[0], As[1]], contr).reshape(Dl, p*p, Dr)
    As2Tenpy.append(A2Site)

As2Dmt = []
for A in AsDmt:
    Dl, p, Dr = A.shape

    contr = ((-1, -2, 1), (1, -3, -4))
    A2Site = ncon([A, A], contr).reshape(Dl, p*p, Dr)
    As2Dmt.append(A2Site)


###############################################################################
# Compare the overlaps based on fidelity density.
###############################################################################

print('Comparing overlaps')
overlaps = []
for i in range(min((len(As2Dmt), len(As2Tenpy)))):
    print(f'Time Step: {tsDmt[i]}')
    overlaps.append(exact_overlap(As2Dmt[i], As2Tenpy[i]))
    print(f'\tOverlap: {overlaps[-1]}')

plt.plot(tsDmt, overlaps, '.')
plt.xlabel('Time')
plt.ylabel('Fidelity Density with iTEBD')
figPath = dmtDataDir / 'compare_itebd.png'

if saveFig:
    plt.savefig(figPath)

plt.show()


###############################################################################
# Compare the overlaps based on trace dist of reduced density matrices.
###############################################################################
from optimise import uniformToRhoN, traceDistance
from analyse import TrAB
from uMPSHelpers import fixedPoints

patchSizes = [2, 4, 8]

data = []
for N in tqdm(patchSizes, desc='N loop'):
    tqdm.write(f'Analysing for N = {N}')
    N2 = N // 2
    patchData = []
    for ADmt, ATenpy in tqdm(zip(As2Dmt, As2Tenpy), total=len(As2Dmt), leave=False):
        rhoDmt = uniformToRhoN(ADmt, N2)
        rhoTenpy = uniformToRhoN(ATenpy, N2)

        patchData.append(traceDistance(rhoDmt, rhoTenpy))
    data.append(patchData)

plt.figure()
for i, N in enumerate(patchSizes):
    patchData = data[i]
    plt.plot(tsDmt, patchData, label=f'Patch Size: {N}')
plt.title('Reduced density trace distance comparison')
plt.xlabel('Time')
plt.ylabel('Trace Dist')
plt.legend()

figPath = dmtDataDir / 'compare_itebd_trace_patch.png'

if saveFig:
    plt.savefig(figPath)

plt.show()
