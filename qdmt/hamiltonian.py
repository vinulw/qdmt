import numpy as np
from functools import reduce
from itertools import product
from numpy import kron, trace, eye
from tqdm import tqdm
import scipy.integrate as integrate
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

if __name__=="__main__":
    g = 0.2
    # H = Hamiltonian({'ZZ':-1, 'X':g}).to_matrix()

    def energySpectrum(k, g):
        return 2*np.sqrt(1+g**2-2*g*np.cos(k))

    # I = integrate.quad(lambda x: energySpectrum(x, g), 0, 2*np.pi)
    # print(I)

    n = 16
    g_range = np.linspace(0.0, 1.6, n)
    Es = np.zeros(n)
    for i, g in tqdm(enumerate(g_range), total=n):
        E = energySpectrum(1, g)
        Es[i] = np.real(E)

    plt.title('Ground state optimisation, exact')
    plt.plot(g_range, Es, label='VUMPS')
    plt.show()
