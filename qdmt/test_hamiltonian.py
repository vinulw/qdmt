import hamiltonian as ham
import numpy as np


def test_TransverseIsing():
    J = -1.
    g = 0.2
    n = 2
    h = ham.TransverseIsing(J, g, n)

    H = ham.Hamiltonian({'ZZ':J, 'X':g}).to_matrix()
    assert np.allclose(h, H)
