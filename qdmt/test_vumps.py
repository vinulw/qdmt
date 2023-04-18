import vumps as v
import numpy as np
from ncon import ncon

def test_random_mixed_gauge():
    d = 2
    D = 4

    AL, AR, C = v.random_mixed_gauge(d, D)

    I = np.eye(D)
    Algauge = ncon([AL, AL.conj()], ((1, 2, -1), (1, 2, -2)))

    assert np.allclose(I, Algauge), 'Al gauge not met'

    Argauge = ncon([AR, AR.conj()], ((-1, 1, 2), (-2, 1, 2)))

    assert np.allclose(I, Argauge), 'Ar gauge not met'

