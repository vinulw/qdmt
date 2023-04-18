import numpy as np
from numpy.linalg import eig
from ncon import ncon

from vumps import random_mixed_gauge

if __name__ == "__main__":
    d = 2
    D = 2
    AL, AR, C = random_mixed_gauge(d, D)

    # AL = np.random.rand(D, d, D) + 1j*np.random.rand(D, d, D)

    ELL = ncon([AL, AL.conj()], ((-1, 1, -3), (-2, 1, -4)))
    ELL = ELL.reshape(D*D, D*D)

    ELL_H = ELL.conj().T

    print(ELL)
    print()
    print(ELL_H)

    w, v = eig(ELL_H)

    evec = v[:, 0]

    print(w[0])
    print('Evec: ')
    print(evec)

    map_evec = ELL_H @ evec
    ratio_evec = map_evec / evec
    print('Map evec')
    print(map_evec)

    print('Ratio evec')
    print(ratio_evec.shape)
    print(ratio_evec)

    print('Verify A v = λ v')
    print(np.allclose(map_evec, w[0] * evec))

    out = ELL_H @ np.eye(D).reshape(-1)
    print(out)

    print(np.allclose(out, np.eye(D).reshape(-1)))

