from circuits import NSiteCircuit_Ansatz_Env, StateAnsatzXZ, SwapTestOps, AddMeasure
import cirq
from cirq.circuits import InsertStrategy
import numpy as np
from itertools import chain

def swap_test_circuit(θA, θB, Q, N, swapQs, Ne=0, ψA=None, ψB=None, Ansatz=StateAnsatzXZ,
                        debug=False):
    '''
    Prepare n qubit SWAP test circuit to compare stateA and stateB which are
    state preparation circuits for ψA and ψB respectively.

    Qubits refer to the qubits to compare on. If None, compare across all the qubits.
    '''
    assert len(Q) == 2*(N + Ne + 1), "Number of qubits should be 2*(N + Ne + 1)"
    QA = Q[:N + Ne + 1]
    QB = Q[N + Ne + 1:]
    if ψA is None:
        ψA = θA

    if ψB is None:
        ψB = θA

    circuitA = NSiteCircuit_Ansatz_Env(θA, ψA, QA, N, Ne, Ansatz).all_operations()
    circuitB = NSiteCircuit_Ansatz_Env(θB, ψB, QB, N, Ne, Ansatz).all_operations()

    circuit = cirq.Circuit()
    circuit.append([circuitA, circuitB])

    swapOps = []
    for swapQ in swapQs:
        swapOps.extend(SwapTestOps(swapQ,))

    if debug:
        circuit.append(swapOps, strategy=InsertStrategy.NEW)
    else:
        circuit.append(swapOps, strategy=InsertStrategy.EARLIEST)


    return circuit

def generate_swapQs(Qs, swapindices):
    assert len(Qs) % 2 == 0, "Need an even number of qubits"

    n = int(len(Qs) / 2)

    QA = Qs[:n]
    QB = Qs[n:]

    return [[QA[i], QB[i]] for i in swapindices]


if __name__=="__main__":
    import itertools
    from simulate import SimulateCircuitLocalNoiseless

    θA = np.random.rand(8)
    θB = np.random.rand(8)

    θA = [0.49839174, 0.06396436, 0.72746455, 0.49554481, 0.09096422,
          0.82718966, 0.81167814, 0.8109978]
    θB = [0.58834982, 0.19390417, 0.83379508, 0.93606346, 0.29246783,
          0.03201088, 0.83285107, 0.27424746]

    N = 3
    Q = cirq.LineQubit.range(2*(N+1))

    swapQs = generate_swapQs(Q, (1, 2))

    circuit = swap_test_circuit(θA, θB, Q, N, swapQs, debug=True)

#    print(swapQs)

    flatQs = list(itertools.chain(*swapQs))
#    print(flatQs)

    QA = [swapQ[0] for swapQ in swapQs]
    QB = [swapQ[1] for swapQ in swapQs]

    bitcount = len(QA)

    circuit = AddMeasure(circuit, QA, 'measA')
    circuit = AddMeasure(circuit, QB, 'measB')


    print(circuit.to_text_diagram(transpose=True))

    res = SimulateCircuitLocalNoiseless(circuit, 10000)

    nprintmax = 10

    measA = res.data['measA'].tolist()
    measB = res.data['measB'].tolist()
    print(res.data)

    bitsA = [cirq.big_endian_int_to_bits(i, bit_count=bitcount) for i in measA]
    bitsB = [cirq.big_endian_int_to_bits(i, bit_count=bitcount) for i in measB]

#    print('Meas A in bits')
#    for a, bita in zip(measA[:nprintmax], bitsA[:nprintmax]):
#        print(f"{a}: {bita}")
#
#    print('Meas B in bits')
#    for b, bitb in zip(measB[:nprintmax], bitsB[:nprintmax]):
#        print(f"{b}: {bitb}")

    def bitwiseAndList(A, B):
        return [int(a == b) for a, b in zip(A, B)]

    bitwiseAndAB = [ bitwiseAndList(a, b) for a, b in zip(bitsA, bitsB)]

    evenparityAB = [(sum(a) + 1) % 2 for a in bitwiseAndAB]

    print("A : B : A & B  : parity")
    for i in range(min((len(bitsA), nprintmax))):
        print(f"{bitsA[i]}  :  {bitsB[i]}  :  {bitwiseAndAB[i]}  :  {evenparityAB[i]}")

    probEven = sum(evenparityAB) / len(evenparityAB)
    print(f"Prob even parity: {probEven}")

