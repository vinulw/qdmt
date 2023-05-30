from vumpt_tools import createMPS, normalizeMPS, mixedCanonical
from vumps import normalise_A
import numpy as np
from numpy import linalg as la
from ncon import ncon


if __name__=="__main__":
    d, D = 2, 8
    print('Testing Al, Ar, C generation')

    # Generation from my code
    myC = np.random.rand(D)
    myC = myC / la.norm(myC)
    myAL = (la.svd(np.random.rand(D* d, D), full_matrices=False)[0]).reshape(D, d, D)
    myAL = normalise_A(myAL)
    myAR = (la.svd(np.random.rand(D* d, D), full_matrices=False)[0]).reshape(D, d, D).transpose(2, 1, 0)
    myAR = normalise_A(myAR)

    # Generation from tutorial

    tutA = createMPS(D, d)
    tutA = normalizeMPS(tutA)
    tutAL, tutAC, tutAR, tutC = mixedCanonical(tutA)

    # Testing the generated As
    I = np.eye(D)
    def testAL(AL):
        ALAL = ncon([AL, AL.conj()], ((1, 2, -1), (1, 2, -2)))
        return np.allclose(I, ALAL)

    def testAR(AR):
        ARAR = ncon([AR, AR.conj()], ((-1, 1, 2), (-2, 1, 2)))
        return np.allclose(I, ARAR)

    print('Testing AL condition...')
    print('   mine: ',  testAL(myAL))
    print('   tut: ', testAL(tutAL))

    print('Testing AR condition...')
    print('   mine: ',  testAR(myAR))
    print('   tut: ', testAR(tutAR))

    def testC(AL, AR, C):

        ALC = ncon([AL, C], ((-1, -2, 1), (1, -3)))
        CAR = ncon([C, AR], ((-1, 1), (1, -2, -3)))

        return np.allclose(ALC, CAR)

    print('Testing C condition...')
    # print('   mine: ', testC(myAL, myAR, myC))
    print('   tut: ', testC(tutAL, tutAR, tutC))
