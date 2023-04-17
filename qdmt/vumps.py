import numpy as np
from numpy import kron, trace, eye
from ncon import ncon
from functools import reduce
from itertools import product

from numpy.linalg import qr
from scipy.sparse.linalg import eigs

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

def left_orthonomalize(A, L0, tol=1e-5):
    '''
    Gauge transform A into left orthonormal form.
    '''
    σ, al, ar = A.shape
    ll, lr = L0.shape

    L = L0/np.linalg.norm(L0)
    Lold = np.copy(L)

    Al, L = qr(ncon([A, L], ((1, -2, -3), (-1, 1))).reshape(σ*ll, -1))

    λ = np.linalg.norm(L)
    L = L / λ

    δ = np.linalg.norm(L - Lold)

    def E_map(Al_curr):
        All, _, Alr = Al_curr.shape
        return ncon([A, Al_curr.conj()], ((-1, 1, -3), (-2, 1,
            -4))).reshape(All*All, Alr*Alr)

    while δ > tol:
        E = E_map(Al)

def largest_evec_left(E):
    '''
    Find leading eigenvector v of E such that vE = λv
    '''
    Eh = E.conj().transpose()

    w, v = eigs(Eh, k=1, which='LM')

    e = v[:, 0]
    e = e.conj().transpose()

    return e

def largest_evec_right(E):
    '''
    Find leading eigenvector v of E such that Ev = λv
    '''
    w, v = eigs(E, k=1, which='LM')

    e = v[:, 0]

    return e



def right_orthonomalize(A, R0, tol):
    '''
    Gauge transform A into left orthonormal form.
    '''
    pass


def mixed_canonical(A, tol):
    '''
    Form a mixed canonical iMPS from a uniform MPS with tensor A.
    '''
    pass

if __name__=="__main__":
    print('Testing evec generator')
    N = 10

    A = np.random.rand(N, N) + 1j * np.random.rand(N, N)

    w, v = eigs(A)

    e = v[:, 0]

    print(w[0])
    print(e)

    Ae = A@e

    λs = Ae / e
    maxL = np.max(λs)
    minL = np.min(λs)
    print(maxL)
    print(minL)
    print(np.allclose(maxL, minL))

    e_func = largest_evec_right(A)
    fun_ratio = e / e_func
    print('Func working: ', np.allclose(fun_ratio, fun_ratio[0]))

    print('Now trying left eigenvector')

    Ah = A.conj().transpose()
    w, v = eigs(Ah)

    e = v[:, 0]
    e = e.conj().transpose()
    print(e)
    print(w[0])

    eA = e@A

    λs = eA / e

    maxL = np.max(λs)
    minL = np.min(λs)
    print(maxL)
    print(minL)
    print(np.allclose(maxL, minL))

    e_func = largest_evec_left(A)
    fun_ratio = e / e_func
    print('Func working: ', np.allclose(fun_ratio, fun_ratio[0]))

