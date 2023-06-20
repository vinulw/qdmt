import numpy as np
from functools import reduce
from itertools import product
from numpy import kron, trace, eye
from tqdm import tqdm
from scipy.integrate import quad
import matplotlib.pyplot as plt

Sx = np.array([[0, 1],
               [1, 0]], dtype=complex)
Sy = np.array([[0, 1j],
               [-1j, 0]], dtype=complex)
Sz = np.array([[1, 0],
               [0, -1]], dtype=complex)
S = {'I': eye(2, dtype=complex), 'X': Sx, 'Y': Sy, 'Z': Sz}

class Hamiltonian:
    """Hamiltonian: string of terms in local hamiltonian.
       Just do quadratic spin 1/2
       ex. tfim = Hamiltonian({'ZZ': -1, 'X': λ}) = Hamiltonian({'ZZ': 1, 'IX': λ/2, 'XI': λ/2})
       for parity invariant specify can single site terms ('X')
       otherwise 'IX' 'YI' etc.

       Credit: Fergus Barratt (qmps/ground_state.py)"""

    def __init__(self, strings=None):
        self.strings = strings
        if strings is not None:
            for key, val in {key:val for key, val in self.strings.items()}.items():
                if len(key)==1:
                    self.strings['I'+key] = val/2
                    self.strings[key+'I'] = val/2
                    self.strings.pop(key)

    def to_matrix(self):
        assert self.strings is not None
        h_i = np.zeros((4, 4))+0j
        for js, J in self.strings.items():
            h_i += J*reduce(kron, [S[j] for j in js])
        self._matrix = h_i
        return h_i

    def from_matrix(self, mat):
        xyz = list(S.keys())
        strings = list(product(xyz, xyz))
        self.strings = {a+b:trace(kron(a, b)@mat) for a, b in strings}
        del self.strings['II']
        return self

def TransverseIsing(J, g, n):
    '''
    Generate an n qubit ising hamiltonian.
    '''

    h = np.zeros((2**n, 2**n)) + 0j

    # Add ZZ terms
    for i in range(n-1):
        pString = ['I'] * n
        pString[i] = 'Z'
        pString[i+1] = 'Z'
        h += J * reduce(kron, [S[j] for j in pString])

    # Add TF term
    for i in range(n):
        pString = ['I'] * n
        pString[i] = 'X'
        h += g / 2. * reduce(kron, [S[j] for j in pString])

    return h

def exact_gs_energy(J, g):
    """
    Exact gs energy for TFIM in the thermodynamic limit.

    For derivation see P. Pfeuty, The one- dimensional Ising model with
    a transverse field, Annals of Physics 57, p. 79 (1970)

    Ref: https://tenpy.readthedocs.io/en/latest/toycode_stubs/tfi_exact.html
    """
    g = -g # Match the two conventions used
    def f(k, lambda_):
        return np.sqrt(1 + lambda_**2 + 2 * lambda_ * np.cos(k))

    E0_exact = -g / (J * 2. * np.pi) * quad(f, -np.pi, np.pi, args=(J / g, ))[0]
    return E0_exact


if __name__=="__main__":
    #J = -1
    #n = 16
    #g_range = np.linspace(0.1, 1.7, n)
    #Es = np.zeros(n)
    #for i, g in tqdm(enumerate(g_range), total=n):
    #    E = exact_gs_energy(J, g)
    #    Es[i] = np.real(E)

    #print('Saving exact gs energy...')
    #data = np.zeros((2, n))
    #data[0] = g_range
    #data[1] = Es
    #header = 'g, energies'
    #np.savetxt('exact_gs_energy.csv', data, delimiter=',')

    print('Loading exact gs_energy...')
    g_range, Es = np.loadtxt('exact_gs_energy.csv', delimiter=',')


    plt.title('Ground state optimisation, exact')
    plt.plot(g_range, Es, label='VUMPS')
    plt.show()
