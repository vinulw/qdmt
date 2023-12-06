import numpy as np
from pathlib import Path
import json
import re
from tqdm import tqdm


dir = Path('./data/28112023-123100/')
logFile = dir / '28112023-123100-evolve_exp.log'
configFile = dir / 'config.json'
print(f'Anlaysing:  {dir}')

with open(configFile, 'r') as cf:
    config = json.load(cf)

testName = config['optConfig']['note']
print(f'Test Name: {testName}')

with open(logFile, 'r') as lf:
    logs = lf.readlines()

dataArray = []
currT = 0.0
print('Loading log...')
for l in tqdm(logs):
    pattern = r'DEBUG - (\b\w+\b)'
    match = re.search(pattern, l)
    # print(l.strip('\n'))
    lineType = match.group(1)

    if lineType == 'Optimisation':
        pattern = 'Optimisation at t=(\d+\.\d+)'
        match = re.search(pattern, l)
        # print(match)
        currT = match.group(1)
        # print(f'Current t: {currT}')
    elif lineType == 'iteration':
        pattern = r'iteration:\t(\d+)\tdist:\t(\d+\.\d+)\tgradient norm:\t(.*)'
        match = re.search(pattern, l)
        # print(f'Match: {match}')

        itNo = match.group(1)
        dist = match.group(2)
        gradNorm = float(match.group(3))
        # print(f'iteration: {itNo}')
        # print(f'dist: {dist}')
        # print(f'gradNorm: {gradNorm}')
        # print(float(gradNorm))

        data = np.array([currT, itNo, dist, gradNorm])
        dataArray.append(data)

    # print()

dataArray = np.array(dataArray)

saveF = dir / 'trainingData.npy'
print(f'Saving file as:  {saveF}')
np.save(saveF, dataArray)
