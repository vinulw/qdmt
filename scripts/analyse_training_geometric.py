import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd

dirs = ['28112023-121951', '28112023-122555', '28112023-123100']
dir = Path(f'./data/{dirs[0]}/')
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

aggTrainDf.plot(subplots=True, logy=True, grid=True,
                title=testName, style=['x--', 'x--'])

saveName = dir / 'training.pdf'
plt.savefig(saveName)
plt.show()
