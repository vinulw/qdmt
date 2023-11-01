'''
Helper functions to evolve uMPS states
'''
import numpy as np
from ncon import ncon

def genAContr(l0, p0, r0, s, N):
    lList = range(l0, s*N+l0, s)
    pList = range(p0, s*N+p0, s)
    rList = range(r0, s*N+r0, s)

    return [list(e) for e in zip(lList, pList, rList)]

def genUContr(topLeft, topRight, botLeft, botRight, stepTop, stepBot, Nu=2):

    topLeft = range(topLeft, topLeft + stepTop*Nu, stepTop)
    topRight = range(topRight, topRight + stepTop*Nu, stepTop)
    botLeft = range(botLeft, botLeft + Nu*stepBot + 1, stepBot)
    botRight = range(botRight, botRight + Nu*stepBot + 1, stepBot)
    return [list(a) for a in zip(topLeft, topRight, botLeft, botRight)]

def firstOrderTrotterEvolve(A, U1, U2, N, l=None, r=None):
    '''
    Evolve density matrix represented by $A$ using a first order trotter
    evolution.

    $$
    \rho_A(t+dt) = (U2 \otimes U2 \otime \dots) (U1 \otimes U1 \otimes \dots)
                    \rho_A(t) (U1^\dagger \otimes U1^\dagger \otimes \dots)
                    (U2^\dagger \otimes U2^\dagger \otimes \dots)
    $$

    And trace out such that only $N$ central sites remain.

        Parameters
        ----------
        A : np.array(D, d, D)
            uMPS state tensor ordered left - physical - right
        U1 : np.array(d**2, d**2)
            First trotter evolution operator
        U2 : np.array(d**2, d**2)
            Second trotter evolution operator
        N : int
            Number of remaining sites for output.

        Returns
        -------
        rhodt : np.array(*[d]*(2*N))
            Time evolved state

    '''
    assert N % 2 == 0, 'N needs to be even'
    D, d, _ = A.shape

    U1ten = U1.reshape(d, d, d, d)
    U2ten = U2.reshape(d, d, d, d)

    # State tensor contractions
    AContr = genAContr(1, 4, 7, 6, N+2)
    ADagContr = genAContr(2, 5, 8, 6, N+2)

    # Get U contr
    evenAContr = genUContr(3, 9, 4, 10, 12, 12, N//2 + 1)
    oddAContr = genUContr(-1, -2, 9, 15, -2, 12, N//2)

    # Get UDagContr
    evenADagContr = genUContr(6, 12, 5, 11, 12, 12, N//2 + 1)
    evenADagContr[0][0] = evenAContr[0][0]
    evenADagContr[-1][1] = evenAContr[-1][1]
    oddADagContr = genUContr(-1-N, -2-N, 12, 18, -2, 12, N//2)
    # l/r contr
    lcontr = [ADagContr[0][0], AContr[0][0]]
    rcontr = [AContr[-1][2], ADagContr[-1][2]]

    # Perform the contraction
    # tensors = [l, r]
    # tensors += [A]*(N+2) + [A.conj()]*(N+2)
    # tensors += [U1ten]*(N//2+1) + [U2ten]*(N//2)
    # tensors += [U1ten.conj()]*(N//2+1) + [U2ten.conj()]*(N//2)
    tensors = [l, r] + [A]*(N+2) + [A.conj()]*(N+2) + [U1ten]*3 + [U2ten]*2 + [U1ten.conj()]*3 + [U2ten.conj()]*2
    contr = [lcontr, rcontr] + AContr + ADagContr + evenAContr
    contr += oddAContr + evenADagContr + oddADagContr

    print(len(tensors))
    print(contr)

    return ncon(tensors, contr)
