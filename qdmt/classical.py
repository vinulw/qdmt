from ncon import ncon
import numpy as np

def unitary_to_tensor_left(U):
    '''
    Take a unitary U and make it a tensor A such that
   |0>     k
    |     |
    |     |     |
    ---U---     | direction of unitary
    |     |     |
    |     |     v
    i     j
    A.shape = (i, j, k)
    i == A == k
         |
         j
    '''
    n = int(np.log2(U.shape[0]))
    zero = np.array([1., 0.])

    Ucontr = list(range(-1, -n-1, -1)) + [1] + list(range(-n-1, -2*n, -1))
    A = ncon([U.reshape(*2 * n *[2]), zero], [Ucontr, [1,]])
    A = A.reshape(2**(n-1), 2, 2**(n-1))
    return A


if __name__=="__main__":
    from circuits import StateAnsatzXZ
    import cirq
    θA = np.random.rand(8)
    θB = np.random.rand(8)

    print(θA)
    print(θB)

    θA = [0.49839174, 0.06396436, 0.72746455, 0.49554481, 0.09096422,
          0.82718966, 0.81167814, 0.8109978]
    θB = [0.58834982, 0.19390417, 0.83379508, 0.93606346, 0.29246783,
          0.03201088, 0.83285107, 0.27424746]

    N = 3
    swap_sites = [0, 1]
    Q = cirq.LineQubit.range(2)

    UA = cirq.unitary(
            StateAnsatzXZ(θA).on(*Q)
        )
    UB = cirq.unitary(
            StateAnsatzXZ(θB).on(*Q)
        )

    A = unitary_to_tensor_left(UA)
    B = unitary_to_tensor_left(UB)

    Aρ = ncon([A, np.conj(A)], ((-1, -3, -5), (-2, -4, -6)))
    i1, i2, σ, l, j1, j2 = Aρ.shape
    Aρ_ = Aρ.reshape(i1*i2, σ, l, j1*j2)
    A1 = Aρ.reshape(i1, i2, σ, l, j1*j2)

    Bρ = ncon([B, np.conj(B)], ((-1, -3, -5), (-2, -4, -6)))
    i1B, i2B, σ, l, j1, j2 = Aρ.shape
    Bρ_ = Bρ.reshape(i1*i2, σ, l, j1*j2)
    B1 = Bρ.reshape(i1, i2, σ, l, j1*j2)

#    Aρ__ = Aρ.reshape(i1*i2, σ, l, j1, j2)
#    Aρ__ = Aρ__.transpose(3, 4, 0, 1, 2)
#    Aρ__ = Aρ__.reshape(j1*j2, i1*i2, σ, l)
#    Aρ__ = Aρ__.transpose(1, 2, 3, 0)
#
#    print("Checking two methods for reshaping are equivalent")
#    print(np.allclose(Aρ_, Aρ__))

    # Calculate contraction for the first site
    if 0 in swap_sites:
        contr = ncon([A1, B1], ((1, 1, 2, 3, -1), (4, 4, 3, 2, -2)))
    else:
        contr = ncon([A1, B1], ((1, 1, 2, 2, -1), (4, 4, 3, 3, -2)))
    print(contr.shape)

    for i in range(1, N+1):
        if i in swap_sites:
            contr = ncon([contr, Aρ_, Bρ_], ((1, 2), (1, 3, 4, -1), (2, 4, 3, -2)))
        else:
            contr = ncon([contr, Aρ_, Bρ_], ((1, 2), (1, 3, 3, -1), (2, 4, 4, -2)))

    zero = np.array([1., 0.], dtype=complex)
    zeroρ = ncon([zero, zero], ((-1,), (-2,)))
    zeroρ = zeroρ.reshape(-1,)
    print("Zero shape: {}".format(zeroρ.shape))

    final = ncon([contr, zeroρ, zeroρ], ((1, 2), (1,), (2,)))

    print("Final overlap: {}".format(final))
