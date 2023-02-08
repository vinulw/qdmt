from swap import *

def test_circuit_generation():
    from circuits import MPOLayerOps, MPSCircuit_Ansatz_Env, StateAnsatzXZ, MPO_Gate_Ops
    from environment import generate_environment_unitary, generate_transferMatrix
    from scipy.stats import unitary_group
    from circuits import MPS_MPO_Circuit, MPS_MPO_Circuit_StateGate
    from circuits import MPS_Circuit_StateGate
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

    print("Verifying state circuit...")
    noQubits = n+2
    qubits = cirq.LineQubit.range(noQubits)
    stateREnvU = generate_environment_unitary(A, W=None, D=D)
    stateREnvGate = cirq.MatrixGate(stateREnvU, name='R1/2')
    circuit = MPS_Circuit_StateGate(stateGate, stateREnvGate, n, qubits)
    print(circuit.to_text_diagram(transpose=True))

    res = SimulateCircuitLocalExact(circuit)
    state = res.state_vector()

    s = np.eye(4)[0]
    Rs = stateREnvU @ s
    Rs = Rs.reshape(D, D)
    ARs = ncon([A, Rs], ((-1, -2, 1), (1, -3)))

    print(np.allclose(ARs.reshape(-1), state))
    print('')

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


if __name__=="__main__":
    test_circuit_generation()
