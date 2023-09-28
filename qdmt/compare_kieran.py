import numpy as np
import pickle

from optimise_state import mixed_state_to_two_site_rho

def trace_dist(A, B):
    assert A.shape == B.shape
    assert len(A.shape) == 2

    dist = A - B
    return np.real(np.trace(dist @ dist.conj().T))

# Load Kieran's psi
fname = 'psi_optimised.pkl'
with open(fname, 'rb') as fle:
    data = pickle.load(fle)

for i in range(3):
    print(i)
    data[i] = data[i].transpose(1, 0, 2)

Al, Ac, Ar, C = data

# Normalise the state found
norm = np.trace(C @ C.conj().T) # normalise state
C = C / np.sqrt(norm)


# Load the rho
rho = np.load('test_rho.npy')

# Calculate optimised rho
rho_opt = mixed_state_to_two_site_rho(Al, C, Ar).reshape(4, 4)
print(trace_dist(rho, rho_opt))
