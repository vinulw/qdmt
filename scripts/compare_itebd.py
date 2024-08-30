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
# Set plotting parameters

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 32

midblue = "#58C4DD"
midred = "#FF8080"
darkblue = "#236B8E"
darkred = "#CF5044"
midgreen = "#A6CF8C"
darkgreen = "#699C52"

###############################################################################
# Compare the output from iTEBD with the DMT algorithm
# - Calculate the trace distance between the two $A$s over a given patch size
#   $N$.
###############################################################################

dmtFname = '16072024-171900'
itebdFname = '17072024-151042'
print('Loading data...')
dmtDataDir = Path(f'./data/{dmtFname}')
iTEBDdata = Path(f'./data/tenpy_timeev/{itebdFname}-Ats.npy')

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

# print('Comparing overlaps')
# overlaps = []
# for i in tqdm(range(min((len(As2Dmt), len(As2Tenpy))))):
#     # print(f'Time Step: {tsDmt[i]}')
#     overlaps.append(exact_overlap(As2Dmt[i], As2Tenpy[i]))
#     # print(f'\tOverlap: {overlaps[-1]}')
#
# plt.plot(tsDmt, overlaps, '.')
# plt.xlabel('Time')
# plt.ylabel('Fidelity Density with iTEBD')
# figPath = dmtDataDir / 'compare_itebd.png'
#
# data = np.array([tsDmt, overlaps]).T
# dataPath = dmtDataDir / 'fidelity_density_itebd_data.csv'
# np.savetxt(dataPath, data, delimiter=',')
#
# if saveFig:
#     plt.savefig(figPath)

###############################################################################
# Compare the overlaps based on trace dist of reduced density matrices.
###############################################################################
from optimise import uniformToRhoN, traceDistance
from analyse import TrAB
from uMPSHelpers import fixedPoints

patchSizes = [4]

lenAs = min(len(As2Dmt), len(As2Tenpy))
As2Dmt = As2Dmt[:lenAs]
As2Tenpy = As2Tenpy[:lenAs]

# data = []
# for N in tqdm(patchSizes, desc='N loop'):
#     tqdm.write(f'Analysing for N = {N}')
#     N2 = N // 2
#     patchData = []
#     for ADmt, ATenpy in tqdm(zip(As2Dmt, As2Tenpy), total=len(As2Dmt), leave=False):
#         rhoDmt = uniformToRhoN(ADmt, N2)
#         rhoTenpy = uniformToRhoN(ATenpy, N2)
#
#         patchData.append(traceDistance(rhoDmt, rhoTenpy))
#     data.append(patchData)
#
# plt.figure()
# for i, N in enumerate(patchSizes):
#     patchData = data[i]
#     plt.plot(tsDmt, patchData, '.', label=f'Patch Size: {N}')
# plt.title('Reduced density trace distance comparison')
# plt.xlabel('Time')
# plt.ylabel('Trace Dist')
# plt.legend()
#
# figPath = dmtDataDir / 'compare_itebd_trace_patch.png'
# header = 't, ' + ', '.join([str(p) for p in patchSizes])
# data = np.array([tsDmt] + data).T
# dataPath = dmtDataDir / 'local_density_itebd_data_4.csv'
# np.savetxt(dataPath, data, delimiter=',', header=header)
#
# if saveFig:
#     plt.savefig(figPath)

###############################################################################
# Trace distance Loschmidt plot
###############################################################################
def traceDistanceN(A, B, N):
    rhoA = uniformToRhoN(A, N)
    rhoB = uniformToRhoN(B, N)

    return traceDistance(rhoA, rhoB)


N2 = 1  # Note that this gets doubled for the final patch size
save = True

method = 'traceDist' # Options are `traceAB` or `traceDist`

dmtLoschPath = dmtDataDir / f'{dmtFname}-{method}-{N2*2}-LoschmidtDmt.npy'
itebdLoschPath = dmtDataDir / f'{itebdFname}-{method}-{N2*2}-LoschmidtItebd.npy'

if not dmtLoschPath.exists() and not itebdLoschPath.exists():
    print('Calculating Loschmidts...')
    A0dmt = As2Dmt[0]
    A0itebd = As2Tenpy[0]

    traceLoschmidtDmt = []
    traceLoschmidtTenpy = []

    if method == 'traceAB':
        traceFunc = TrAB
    else:
        traceFunc = traceDistanceN

    print('Calculating trace dist Loschmidt...')
    for Admt, Atenpy in tqdm(zip(As2Dmt, As2Tenpy), total=len(As2Dmt)):
        traceLoschmidtDmt.append(traceFunc(A0dmt, Admt, N2))
        traceLoschmidtTenpy.append(traceFunc(A0itebd, Atenpy, N2))

    if save:
        np.save(dmtLoschPath, traceLoschmidtDmt)
        np.save(itebdLoschPath, traceLoschmidtTenpy)

    traceLoschmidtDmt = np.array(traceLoschmidtDmt)
    traceLoschmidtTenpy = np.array(traceLoschmidtTenpy)
else:
    print('Loading Loschmidts...')
    print(f'iTEBD Data: {itebdLoschPath}')
    print(f'DMT Data: {dmtLoschPath}')
    traceLoschmidtDmt = np.load(dmtLoschPath)
    traceLoschmidtTenpy = np.load(itebdLoschPath)

if method == 'traceDist':
    traceLoschmidtDmt_ = 1 - traceLoschmidtDmt
    traceLoschmidtTenpy_ = 1 - traceLoschmidtTenpy
else:
    traceLoschmidtDmt_ = traceLoschmidtDmt
    traceLoschmidtTenpy_ = traceLoschmidtTenpy

traceLoschmidtDmt_ = -1*np.log(np.abs(traceLoschmidtDmt_))
traceLoschmidtTenpy_ = -1*np.log(np.abs(traceLoschmidtTenpy_))

if method == 'traceAB':
    traceLoschmidtDmt_ /= (N2*2)
    traceLoschmidtTenpy_ /= (N2*2)

print('Finished Loschmidts...\nPlotting...')

plt.figure(figsize=(12, 6))

plt.plot(tsDmt, traceLoschmidtDmt_, ls='--', marker='x', label='Local Cost', color=darkgreen)
plt.plot(tsDmt, traceLoschmidtTenpy_, ls='--', marker='o', fillstyle='none', label='iTEBD', color=darkblue)

# plt.title('Local trace distance Loschmidt')
plt.xlabel('Time')
if method == 'traceDist':
    plt.ylabel('Trace Distance Loschmidt')
else:
    plt.ylabel('Trace Loshcmidt Echo')
plt.legend(fontsize=24)
plt.grid()
plt.tight_layout()

savePath = dmtDataDir / f'{method}_{N2*2}_Losch.png'
plt.savefig(savePath)

if method == 'traceDist':
    plt.figure(figsize=(12, 6))
    plt.plot(tsDmt, traceLoschmidtDmt, ls='--', marker='x', label='Local Cost', color=darkgreen)
    plt.plot(tsDmt, traceLoschmidtTenpy, ls='--', marker='o', fillstyle='none', label='iTEBD', color=darkblue)

    plt.xlabel('Time')
    plt.ylabel(r'$Tr(\rho(t) - \rho(0))^2$')
    plt.legend(fontsize=24)
    plt.grid()
    plt.title('Trace distance with initial over evolution')
    plt.tight_layout()

    plt.figure(figsize=(12, 6))
    diffTraceLoschmidt = np.abs(traceLoschmidtTenpy - traceLoschmidtDmt)
    plt.plot(tsDmt, diffTraceLoschmidt, ls='--', marker='x', color=darkred)

    plt.xlabel('Time')
    plt.ylabel(r'$|D_{itebd} - D_{local}|$')
    plt.legend(fontsize=24)
    plt.grid()
    plt.title('Difference in trace dist with initial over evolution')
    plt.tight_layout()

plt.show()
