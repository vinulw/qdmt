'''
This script prepares a ground state of the Transverse Field Ising model and
performs a quench across the quantum critical point.
'''
import sys
sys.path.insert(0, '../qdmt/')

import numpy as np
import scipy.linalg as la
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from tqdm import tqdm
import contextlib
from os import path, makedirs
import os
import json

from datetime import datetime

from hamiltonian import TransverseIsing
from vumps import vumps
from evolve import firstOrderTrotterEvolveTransposed
from optimise import optimiseDensityGradDescent
from optimise import gradDescentGrassmannCanonical
from logOutput import OutputLogger


save_dir  = './data/16072024-171900'
now ='16072024-171900'
###############################################################################
# Prepare some hyper parameters
###############################################################################

configPath = path.join(save_dir, 'config.json')
with open(configPath, 'r') as f:
    config = json.load(f)
    g1 = config['g1']
    g2 = config['g2']

    dt = config['dt']
    maxTime = config['maxTime']
    D = config['D']
    d = config['d']
    N = config['N']
    optConfig = config['optConfig']

saveAs = True

times = [float(f[3:-4].replace('-', '.')) for f in os.listdir(save_dir) if f[:2] == "A_"]

currTime = max(times)
print(maxTime)

print(f'New time range: {currTime + dt} → {maxTime}')
tRange = np.arange(currTime + dt, maxTime, dt)

print("Loading most recent A...")
Afname = f'A_t{currTime:.02f}'.replace('.', '-') + '.npy'
APath = path.join(save_dir, Afname)
print(APath)

A0 = np.load(APath)
print('Loaded A0')

assert False
###############################################################################
# Perform the time evolution
###############################################################################

print('Initialising time evolution...')
h2 = TransverseIsing(1, g2, 2)
U = la.expm(-1j*h2*dt).reshape(d, d, d, d)


Ats = [A0]
errors = []
At = A0

print('Performing time evolution...')

# Prepare logger
now = datetime.now().strftime('%d%m%Y-%H%M%S') # For file saving
fname = f'{now}-evolve_exp.log'
fname = path.join(save_dir, fname)
format = '%(asctime)s - %(levelname)s - %(message)s'
logger = OutputLogger('evLogger', 'DEBUG', fname=fname, format=format)

for t in tqdm(tRange[1:]):
    # Send output of optimisation to log file
    with contextlib.redirect_stdout(logger):
        print(f'Optimisation at t={t:.3f}')
        rhotdt = firstOrderTrotterEvolveTransposed(At, U, U, N) # Evolve the state
        error, Atdt = gradDescentGrassmannCanonical(
            rhotdt, D, eps=optConfig['eps'], A0=At,
            tol=optConfig['tol'], maxIter=optConfig['maxIter'])
        # error, Atdt = optimiseDensityGradDescent(
        #     rhotdt, D, eps=optConfig['eps'], A0=At,
        #     tol=optConfig['tol'], maxIter=optConfig['maxIter'])

    Ats.append(Atdt)
    errors.append(error)

    At = Atdt
    if saveAs:
        fname = path.join(save_dir, f'A_t{t:.02f}'.replace('.', '-') + '.npy')
        np.save(fname, At)

logging.basicConfig(level=logging.DEBUG)    # Reset the logging

print(f'Written output to: {save_dir}')
