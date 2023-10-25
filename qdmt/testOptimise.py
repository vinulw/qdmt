from uMPSHelpers import createMPS, normalizeMPS
from optimise import uniformToRho, gradient
import numpy as np

def testGradient():
    d, D = 2, 4
    A = createMPS(D, d)
    A = normalizeMPS(A)

    rhoA = uniformToRho(A)
    grad = gradient(rhoA, A)

    assert np.allclose(np.linalg.norm(grad), 0)



