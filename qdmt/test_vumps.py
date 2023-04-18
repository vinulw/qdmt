import vumps as v
import numpy as np
from ncon import ncon

def test_random_mixed_gauge():
    d = 2
    D = 4

    AL, AR, C = v.random_mixed_gauge(d, D)
    C = np.diag(C)

    I = np.eye(D)
    Algauge = ncon([AL, AL.conj()], ((1, 2, -1), (1, 2, -2)))

    assert np.allclose(I, Algauge), 'Al gauge not met'

    Argauge = ncon([AR, AR.conj()], ((-1, 1, 2), (-2, 1, 2)))

    assert np.allclose(I, Argauge), 'Ar gauge not met'

    AC_L = ncon([AL, C], ((-1, -2, 1), (1, -3)))
    AC_R = ncon([C, AR], ((-1, 1), (1, -2, -3)))

    print('Verifying gauge')
    print(np.allclose(AC_L, AC_R))

    print(AC_L)
    print(AC_R)

if __name__=="__main__":
    test_random_mixed_gauge()
