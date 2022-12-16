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


if __name__=="__main__":
    from circuits import MPOLayerOps, MPSCircuit_Ansatz_Env, StateAnsatzXZ, MPO_Gate_Ops
    from environment import generate_environment_unitary, generate_transferMatrix
    from scipy.stats import unitary_group
    from circuits import MPS_MPO_Circuit, MPS_MPO_Circuit_StateGate
    from ncon import ncon
    from simulate import SimulateCircuitLocalExact

    d= 2
    D = 2
    θ = np.random.randn(8)
    ψ = np.random.randn(8)
    n = 1
    ne = 0
    offset=3
    noQubits = n+4
    noQubits = 3
    qubits = cirq.LineQubit.range(noQubits)

    stateU = unitary_group.rvs(d*D)
    A = stateU.reshape(d, D, d, D)
    A = A.transpose(2, 3, 0, 1)
    zero_state = np.eye(d)[0, :]
    stateGate = cirq.MatrixGate(stateU.T, name='U')

    print('Checking state generation')
    circuit = cirq.Circuit()
    circuit.append(stateGate.on(qubits[1], qubits[2]))
    circuit.append(stateGate.on(qubits[0], qubits[1]))
    print(circuit.to_text_diagram(transpose=True))

    res = SimulateCircuitLocalExact(circuit)
    state = res.state_vector()

    A = ncon([A, zero_state], ((-1, -2, 1, -3), (1,))) # (Dl, σ, Dr)
    contr = ncon([A, A, zero_state], ((-1, -2, 1), (1, -3, 2), (2,)))
    contr = contr.reshape(-1,)
#    print('Classical Simulation')
#    print(contr)
#    print('Circuti Simulation')
#    print(state)
    compare = np.allclose(contr, state)
    print(f'Classical == Circuit: {compare}')
    print()

    print('Checking operator generation')
    mpoU = unitary_group.rvs(d*D)
    W = mpoU.reshape(D, d, d, D) # (Dl, σ, l, Dr)
    W = W.transpose(1, 2, 0, 3) # (σ, l, Dl, Dr)
    mpoGate = cirq.MatrixGate(mpoU, name='W')

    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    ops = MPO_Gate_Ops(mpoGate, 0, 1, qubits)
    circuit.append(ops)
#    circuit.append(mpoGate.on(*qubits))
    print(circuit.to_text_diagram(transpose=True))

    res = SimulateCircuitLocalExact(circuit)
    state = res.state_vector()

    contr = ncon([W, zero_state, zero_state], ((-2, 1, -1, 2), (1,), (2,)))
    contr = contr.reshape(-1,)

    compare = np.allclose(contr, state)
    print(f'Classical == Circuit: {compare}')
    print()

    print('Checking environment generation')
    rightEnvironmentU = generate_environment_unitary(A, W=W, D=D)
    s = np.eye(d**2*D**2)[0]
    R1_2 = rightEnvironmentU @ s
    R1_2 = R1_2.reshape(D*D, -1)
    envU = ncon([R1_2, R1_2.conj()], ((-1, 1), (-2, 1)))
    envU = envU.reshape(D, D, D, D)
    envU = envU.transpose(0, 1, 3, 2)
#    envU = envU.reshape(-1)

    rightEnvironmentGate = cirq.MatrixGate(rightEnvironmentU, name='R1/2')

    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(4)
    circuit.append(rightEnvironmentGate.on(*qubits))
    print(circuit.to_text_diagram(transpose=True))

    res = SimulateCircuitLocalExact(circuit)
    state = res.state_vector()
    R1_2 = R1_2.reshape(-1,)
    compare = np.allclose(R1_2, state)
    print(f'Classical == Circuit: {compare}')
    print()

    print('Verifying AR circuit')
    qubits = cirq.LineQubit.range(5)
    circuit = cirq.Circuit()
    circuit.append(rightEnvironmentGate.on(*qubits[1:]))
    circuit.append(stateGate.on(*qubits[:2]))
    print(circuit.to_text_diagram(transpose=True))
    res = SimulateCircuitLocalExact(circuit)
    state = res.state_vector()

    R = R1_2.reshape(D, D, D, D)
    AR = ncon([A, R], ((-1, -2, 1), (1, -3, -4, -5)))

    print(np.allclose(state, AR.reshape(-1)))

    print('Verifying AWR circuit')
    qubits = cirq.LineQubit.range(5)
    circuit = cirq.Circuit()
    circuit.append(rightEnvironmentGate.on(*qubits[1:]))
    circuit.append(stateGate.on(*qubits[:2]))
    circuit.append(mpoGate.on(*qubits[1:3]))
    print(circuit.to_text_diagram(transpose=True))
    res = SimulateCircuitLocalExact(circuit)
    state = res.state_vector()

    R = R1_2.reshape(D, D, D, D)
    AWR = ncon([A, W, R], ((-1, 3, 1), (-3, 3, -2, 2),  (1, 2, -4, -5)))

    print(np.allclose(state, AWR.reshape(-1)))

    print('Verifying full circuit')
    noQubits = n+4
    qubits = cirq.LineQubit.range(noQubits)
    circuit = MPS_MPO_Circuit_StateGate(stateGate, mpoGate, rightEnvironmentGate, n, qubits)
    print(circuit.to_text_diagram(transpose=True))

    circuitDagger = cirq.inverse(circuit)
#    circuit.append(circuitDagger)

    AWWA = ncon([A, W, W.conj(), A.conj()],
            ((-1, 1, -5), (2, 1, -2, -6), (2, 3, -3, -7), (-4, 3, -8)))
    # AWWA.shape = (Dl_A, Dl_W, Dl_Wconj, Dl_Aconj, Dr_A, Dr_W, Dr_Wconj, Dr_Aconj)
    IAWWA = ncon([AWWA, ], ((1, 2, 2, 1, -1, -2, -3, -4), ))
    IAWWAR = ncon([IAWWA, envU], ((1, 2, 3, 4), (1, 2, 3, 4)))

    AWWAR = ncon([AWWA, envU], ((-1, -2, -3, -4, 1, 2, 3, 4), (1, 2, 3, 4)))
    print(IAWWAR)

    R = R1_2.reshape(D, D, D, D)
    AWR = ncon([A, W, R], ((-1, 1, 2), (-3, 1, -2, 3), (2, 3, -4, -5)))
    AWR = AWR.reshape(-1, )
    print(AWR.shape)

    res = SimulateCircuitLocalExact(circuit)
    state = res.state_vector()
    print(state.shape)

    print(np.allclose(AWR, state))
#    print(AWR)
#    print()
#    print(state)


