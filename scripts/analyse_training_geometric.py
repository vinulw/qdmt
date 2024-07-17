import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd

dirs = ['28112023-121951', '28112023-122555', '28112023-123100']
markers = ['o', '^', 'x']

plt.rcParams.update({
    'font.size': 20,
    'legend.fontsize': 12
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

for i, d in enumerate(dirs):
    m = markers[i % len(markers)]
    dir = Path(f'./data/{d}/')
    trainFile = dir / 'trainingData.npy'
    configFile = dir / 'config.json'

    with open(configFile, 'r') as cf:
        config = json.load(cf)

    testName = config['optConfig']['note']
    print(f'Test Name: {testName}')

    trainData = np.load(trainFile)

    cols = ['time', 'iteration', 'tDist', 'gradNorm']
    trainDf = pd.DataFrame(trainData, columns=cols)
    typeDict = {
        'time': 'float',
        'iteration': 'int',
        'tDist': 'float',
        'gradNorm': 'float'
    }
    trainDf = trainDf.astype(typeDict)

    aggTrainDf = trainDf.groupby('iteration')
    aggTrainDf = aggTrainDf[['tDist', 'gradNorm']].mean()

    # aggTrainDf.plot(subplots=True, logy=True, grid=True,
    #                 title=testName, style=['x--', 'x--'])

    aggTrainDf.plot(y='tDist', ax=ax1, label=testName, style=f'{m}--', logy=True)
    aggTrainDf.plot(y='gradNorm', ax=ax2, label=testName, style=f'{m}--', logy=True)

ax1.set_xlabel('')
ax1.grid(True)
ax1.set_ylabel('Trace Distance')
ax2.set_ylabel('Gradient Norm')
ax2.grid(True)
plt.legend()
saveName = 'comparingRiemannian.pdf'
plt.savefig(saveName)
plt.show()
