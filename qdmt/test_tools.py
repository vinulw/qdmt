import numpy as np
from tools import createMPS, mixedCanonical
from ncon import ncon

def test_mixedCanonical():
    D = 8
    d = 2
    A = createMPS(D, d)

    Al, Ar, C = mixedCanonical(A)

    I = np.eye(D)
    vZeros = np.isclose(I.reshape(-1), 0)
    notvZeros = np.invert(vZeros)

    # Checking Al condition
    ALAL = ncon([Al, Al.conj()], ((1, 2, -1), (1, 2, -2)))
    λs = ALAL.reshape(-1)[notvZeros] / I.reshape(-1)[notvZeros]
    assert np.allclose(λs, λs[0])

    # Checking Ar condition
    ARAR = ncon([Ar, Ar.conj()], ((-1, 1, 2), (-2, 1, 2)))
    λs = ARAR.reshape(-1)[notvZeros] / I.reshape(-1)[notvZeros]
    assert np.allclose(λs, λs[0])

    # Gauge condition
    CAR = ncon([C, Ar], ((-1, 1), (1, -2, -3)))
    ALC = ncon([Al, C], ((-1, -2, 1), (1, -3)))

    assert np.allclose(CAR, ALC)

