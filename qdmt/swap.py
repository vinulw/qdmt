from circuits import MPSCircuit_Ansatz_Env, StateAnsatzXZ, SwapTestOps, AddMeasure
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

    circuitA = MPSCircuit_Ansatz_Env(θA, ψA, QA, N, Ne, Ansatz).all_operations()
    circuitB = MPSCircuit_Ansatz_Env(θB, ψB, QB, N, Ne, Ansatz).all_operations()

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

def main_old():
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


if __name__ == "__main__":
    from circuits import StateAnsatzXZ, MPS_Circuit_StateGate
    from ncon import ncon
    from environment import generate_environment_unitary
    from environment import generate_state_transferMatrix, generate_right_environment
    from dmt_overlap import partial_density_matrix

    from cirq.circuits import InsertStrategy
    from simulate import SimulateCircuitLocalNoiseless, SimulateCircuitLocalExact

    θA = np.random.rand(8)
    θB = np.random.rand(8)
    d = 2
    D = 2
    n = 3
    qubits = cirq.LineQubit.range(2*(n+2))

    zeroState = np.eye(d)[0, :]

    stateGateA = StateAnsatzXZ(θA)
    stateGateA.set_name('A')
    stateGateB = StateAnsatzXZ(θB)
    stateGateB.set_name('B')

    stateUnitaryA = cirq.unitary(stateGateA)
    stateUnitaryB = cirq.unitary(stateGateB)

    A = stateUnitaryA.T.reshape(d, D, d, D)
    A = A.transpose(2, 3, 0, 1)
    A = ncon([A, zeroState], ((-1, -2, 1, -3), (1,))) # (Dl, σ, Dr)

    B = stateUnitaryB.T.reshape(d, D, d, D)
    B = B.transpose(2, 3, 0, 1)
    B = ncon([B, zeroState], ((-1, -2, 1, -3), (1,))) # (Dl, σ, Dr)

    rightEnvironmentUA = generate_environment_unitary(A, D=D)
    rightEnvironmentGateA = cirq.MatrixGate(rightEnvironmentUA, name='RA')
    s = np.eye(4)[0]
    RA = rightEnvironmentUA @ s
    RA = RA.reshape(D, D)

    rightEnvironmentUB = generate_environment_unitary(B, D=D)
    rightEnvironmentGateB = cirq.MatrixGate(rightEnvironmentUB, name='RB')
    RB = rightEnvironmentUB @ s
    RB = RB.reshape(D, D)

    circuitA = MPS_Circuit_StateGate(stateGateA, rightEnvironmentGateA, n, qubits[:n+2])
    circuitB = MPS_Circuit_StateGate(stateGateB, rightEnvironmentGateB, n, qubits[n+2:])

    circuit = cirq.Circuit()
    circuit.append([circuitA.all_operations(), circuitB.all_operations()])

    swapQs = generate_swapQs(qubits, (2,))
    print(swapQs)
    nprintmax = 10
    for swapQ in swapQs:
        swapOps = SwapTestOps(swapQ)
        circuit.append(swapOps, strategy=InsertStrategy.NEW)

    QA = [swapQ[0] for swapQ in swapQs]
    QB = [swapQ[1] for swapQ in swapQs]

    circuit = AddMeasure(circuit, QA, 'MA')
    circuit = AddMeasure(circuit, QB, 'MB')

    Nshots = 10000
    print(circuit.to_text_diagram(transpose=True))
    res = SimulateCircuitLocalNoiseless(circuit, Nshots)

    measA = res.data['MA'].tolist()
    measB = res.data['MB'].tolist()

    bitcount = len(QA)
    bitsA = [cirq.big_endian_int_to_bits(i, bit_count=bitcount) for i in measA]
    bitsB = [cirq.big_endian_int_to_bits(i, bit_count=bitcount) for i in measB]

    def bitwiseAndList(A, B):
        return [int(a == b) for a, b in zip(A, B)]

    bitwiseAndAB = [ bitwiseAndList(a, b) for a, b in zip(bitsA, bitsB)]

    evenparityAB = [(sum(a) + 1) % 2 for a in bitwiseAndAB]

    print(f"{'A': ^5}:{'B': ^7}:{'A & B': ^7}: parity")
    for i in range(min((len(bitsA), nprintmax))):
        print(f"{bitsA[i]}  :  {bitsB[i]}  :  {bitwiseAndAB[i]}  :  {evenparityAB[i]}")

    probEven = sum(evenparityAB) / len(evenparityAB)
    traceCircuit = 2*probEven - 1

    print(f"Prob even parity: {probEven}")
    print(f"Trace on circuit: {traceCircuit}")

    pFailure = 0
    for a, b in zip(bitsA, bitsB):
        if a[0] == 1 and b[0] == 1:
            pFailure += 1
    pFailure = pFailure / Nshots
    print(f"Prob failure: {pFailure}")
    pSuccess = 1.0 - pFailure
    traceCircuit = 2*pSuccess - 1
    print(f"Trace on circuit: {traceCircuit}")

    # Calculating the trρAρB classically
    ρA = partial_density_matrix(A, RA, N=3, ignored_indices=[1], halfEnv=True)
    ρB = partial_density_matrix(B, RB, N=3, ignored_indices=[1], halfEnv=True)

    print('Measuring trρAρB...')

    trρAρB = ncon([ρA, ρB], ((1, 2), (2, 1)))
    print(trρAρB)

    AAAR = ncon([A, A, A, RA], ((-1, -2, 1), (1, -3, 2), (2, -4, 3), (3, -5)))

    print('Verifying state A generated...')
    res = SimulateCircuitLocalExact(circuitA)
    state = res.state_vector()
    state = state.reshape(*[2]*(n+2))
    print(np.allclose(AAAR, state))

    contr0 = list(range(-1, -2*(n+2)-1, -2))
    contr1 = list(range(-2, -2*(n+2)-1, -2))

    traced = (0, 1, 3, 4)
    i = 1
    for t in traced:
        contr0[t] = i
        contr1[t] = i
        i+=1

    print('Checking ρCircuitA == ρA')
    ρCircuitA = ncon([state, state.conj()], (contr0, contr1))

    print(np.allclose(ρA, ρCircuitA))

    # Checking ρB
    circuitB = MPS_Circuit_StateGate(stateGateB, rightEnvironmentGateB, n, qubits[:n+2])
    # print(circuitB.to_text_diagram(transpose=True))
    BBBR = ncon([B, B, B, RB], ((-1, -2, 1), (1, -3, 2), (2, -4, 3), (3, -5)))
    res = SimulateCircuitLocalExact(circuitB)
    state = res.state_vector()
    state = state.reshape(*[2]*(n+2))
    print('Verifying state B is close...')
    print(np.allclose(BBBR, state))
    ρCircuitB = ncon([state, state.conj()], (contr0, contr1))

    print('Checking ρCircuitB == ρB')
    print(np.allclose(ρB, ρCircuitB))

    from dmt_overlap import trace_distance

    print("Trace distance ρCircuitB, ρB")
    print(trace_distance(ρCircuitB, ρB))
