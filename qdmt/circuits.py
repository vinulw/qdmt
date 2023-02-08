import cirq
import numpy as np
from numpy.lib.function_base import append
from scipy.linalg import expm
from ground_state import Hamiltonian
from cirq import two_qubit_matrix_to_sqrt_iswap_operations

###################################
# QMPS Circuit Ansatz
###################################

class StateAnsatzXZ(cirq.Gate):
    def __init__(self, Psi):
        self.Psi = Psi
        self.name='U'

    def _decompose_(self, qubits):
        return [
                cirq.rz(self.Psi[0]).on(qubits[0]),
                cirq.rx(self.Psi[1]).on(qubits[0]),
                cirq.rz(self.Psi[2]).on(qubits[1]),
                cirq.rx(self.Psi[3]).on(qubits[1]),
                cirq.CNOT(*qubits),
                cirq.rz(self.Psi[4]).on(qubits[0]),
                cirq.rx(self.Psi[5]).on(qubits[0]),
                cirq.rz(self.Psi[6]).on(qubits[1]),
                cirq.rx(self.Psi[7]).on(qubits[1]),
                cirq.CNOT(*qubits),
#                cirq.rz(self.Psi[8]).on(qubits[0]),
#                cirq.rx(self.Psi[9]).on(qubits[0]),
#                cirq.rz(self.Psi[10]).on(qubits[1]),
#                cirq.rx(self.Psi[11]).on(qubits[1]),
#                cirq.CNOT(*qubits)
        ]

    def set_name(self, name):
        assert type(name) == str
        self.name = name

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return [self.name, self.name]

class StateAnsatzKak(cirq.Gate):
    def __init__(self, Psi):
        assert len(Psi) == 15
        self.Psi = Psi

    def _decompose_(self, qubits):
        return [
                cirq.rz(self.Psi[0]).on(qubits[0]),
                cirq.ry(self.Psi[1]).on(qubits[0]),
                cirq.rz(self.Psi[2]).on(qubits[0]),

                cirq.rz(self.Psi[3]).on(qubits[1]),
                cirq.ry(self.Psi[4]).on(qubits[1]),
                cirq.rz(self.Psi[5]).on(qubits[1]),

                cirq.XX(*qubits)**self.Psi[6],
                cirq.YY(*qubits)**self.Psi[7],
                cirq.ZZ(*qubits)**self.Psi[8],

                cirq.rz(self.Psi[9]).on(qubits[0]),
                cirq.ry(self.Psi[10]).on(qubits[0]),
                cirq.rz(self.Psi[11]).on(qubits[0]),

                cirq.rz(self.Psi[12]).on(qubits[1]),
                cirq.ry(self.Psi[13]).on(qubits[1]),
                cirq.rz(self.Psi[14]).on(qubits[1]),
        ]

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ['U','U']

class ShallowFullStateAnsatz(cirq.Gate):
    def __init__(self, Psi):
        assert len(Psi) == 15
        self.Psi = Psi

    def _decompose_(self, qubits):
        return [
            cirq.rz(self.Psi[0]).on(qubits[0]),
            cirq.rx(self.Psi[1]).on(qubits[0]),
            cirq.rz(self.Psi[2]).on(qubits[0]),

            cirq.rz(self.Psi[3]).on(qubits[1]),
            cirq.rx(self.Psi[4]).on(qubits[1]),
            cirq.rz(self.Psi[5]).on(qubits[1]),

            cirq.CNOT(qubits[0], qubits[1]),

            cirq.ry(self.Psi[6]).on(qubits[0]),

            cirq.CNOT(qubits[1], qubits[0]),

            cirq.ry(self.Psi[7]).on(qubits[0]),
            cirq.rz(self.Psi[8]).on(qubits[1]),

            cirq.CNOT(qubits[0], qubits[1]),

            cirq.rz(self.Psi[9]).on(qubits[0]),
            cirq.rx(self.Psi[10]).on(qubits[0]),
            cirq.rz(self.Psi[11]).on(qubits[0]),

            cirq.rz(self.Psi[12]).on(qubits[1]),
            cirq.rx(self.Psi[13]).on(qubits[1]),
            cirq.rz(self.Psi[14]).on(qubits[1]),
        ]

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ['U','U']


def TimeEvolutionOperations(g1, dtau):
    q0, q1 = cirq.LineQubit.range(2)
    U = expm(-1j*Hamiltonian({'ZZ':-1, 'X':g1}).to_matrix()*dtau*2.0)

    ops = two_qubit_matrix_to_sqrt_iswap_operations(q0, q1, U, clean_operations=True)
    return [q0, q1], ops

def TimeEvolutionOpLayer(g1, dtau, qubits):
    t_qubits, Uops = TimeEvolutionOperations(g1, dtau)

    ops = []
    for i in range(1, len(qubits)-1, 2):
        q0, q1 = qubits[i], qubits[i+1]
        mapdict = {t_qubits[0]: q0,
                t_qubits[1]: q1}
        U_ = [U.transform_qubits(mapdict) for U in Uops]
        ops.append(U_)

    return ops

def MPOLayerOps(AnsatzOps, AnsatzQubits, CircuitQubits):
    ops = []
    for i in range(1, len(CircuitQubits), 2):
        q0, q1 = CircuitQubits[i], CircuitQubits[i+1]
        mapdict = {
                AnsatzQubits[0] : q0,
                AnsatzQubits[1] : q1
                }
        CurrentOps = [Op.transform_qubits(mapdict) for Op in AnsatzOps]
        ops.append(CurrentOps)
    return ops



##############################################################
# MPS Circuits
##############################################################

def MPSCircuit_Ansatz_Env(θ, ψ, Q, N, Ne, Ansatz=StateAnsatzXZ, offset=0):

    c = cirq.Circuit()

    for i in range(Ne):
        c.append(cirq.decompose_once( Ansatz(ψ).on( Q[-2-i-offset], Q[-1-i-offset] )))

    for i in range(N):
        c.append(cirq.decompose_once( Ansatz(θ).on( Q[-2-i-Ne-offset], Q[-1-i-Ne-offset] ) ) )

    return c

def MPSCircuit_Ansatz_Env_Right(θ, ψ, Q, N, Ne, Ansatz=StateAnsatzXZ, offset=0):

    c = cirq.Circuit()

    for i in range(Ne):
        c.append(cirq.decompose_once( Ansatz(ψ).on( Q[i], Q[i+1] )))

    for i in range(N):
        c.append(cirq.decompose_once( Ansatz(θ).on( Q[i+Ne+offset], Q[i+1+Ne+offset] ) ) )

    return c

def MPO_Gate_Ops(mpoGate, startIndex: int, stopIndex: int, Qubits):
    assert startIndex <= stopIndex, 'Start index must be lower than stop index'

    ops = []
    for i in reversed(range(startIndex, stopIndex)):
        ops.append(mpoGate.on(Qubits[i], Qubits[i+1]))

    return ops

def MPS_MPO_Circuit(θ, MPOGate, rightEnvGate, stateAnsatz, N, Qubits=None):
    if Qubits is None:
        Qubits = cirq.LineQubit.range(N+4)
    noQubits = len(Qubits)
    offset=3
    assert noQubits >= N+4, "Not enough qubits"

    # Add environment
    circuit = cirq.Circuit()
    circuit.append(rightEnvGate.on(*Qubits[-4:]))

    # Add MPS
    circuitmps = MPSCircuit_Ansatz_Env(θ, None, Qubits, N, 0, stateAnsatz, offset=offset)
    circuit.append(circuitmps)

    # Add MPO
    startIndex = 1
    endIndex = startIndex + N
    ops = MPO_Gate_Ops(MPOGate, startIndex, endIndex, Qubits)
    circuit.append(ops)

    return circuit

def MPS_MPO_Circuit_StateGate(StateGate, MPOGate, rightEnvGate, N, Qubits=None):
    if Qubits is None:
        Qubits = cirq.LineQubit.range(N+4)
    noQubits = len(Qubits)
    offset=3
    assert noQubits >= N+4, "Not enough qubits"

    # Add environment
    circuit = cirq.Circuit()
    circuit.append(rightEnvGate.on(*Qubits[-4:]))

    # Add MPS
    for i in reversed(range(N)):
        q0, q1 = Qubits[i], Qubits[i+1]
        circuit.append(StateGate.on(q0, q1))

    # Add MPO
    startIndex = 1
    endIndex = startIndex + N
    ops = MPO_Gate_Ops(MPOGate, startIndex, endIndex, Qubits)
    circuit.append(ops)

    return circuit

def MPS_Circuit_StateGate(StateGate, rightEnvGate, N, Qubits=None):
    if Qubits is None:
        Qubits = cirq.LineQubit.range(N+2)
    noQubits = len(Qubits)
    offset = 1
    assert noQubits >= N+2, "Not enough qubits"

    # Add environment
    circuit = cirq.Circuit()
    circuit.append(rightEnvGate.on(*Qubits[-2:]))

    # Add MPS
    for i in reversed(range(N)):
        q0, q1 = Qubits[i], Qubits[i+1]
        circuit.append(StateGate.on(q0, q1))

    return circuit


def OverlapCircuitEnv(θA, θB, Q, N, Ne=0, ψA=None, ψB=None,
                      Ansatz=StateAnsatzXZ, offset=0):
    ''''
    Generate overlap circuits with environments.
    '''
    if ψA is None:
        ψA = θA

    if ψB is None:
        ψB = θA

    circuitA = NSiteCircuit_Ansatz_Env(θA, ψA, Q, N, Ne, Ansatz, offset)
    circuitB = NSiteCircuit_Ansatz_Env(θB, ψB, Q, N, Ne, Ansatz, offset)

    circuit = cirq.Circuit()
    circuit.append(circuitA)
    circuit.append(cirq.inverse(circuitB))
    return circuit


def OverlapCircuitEnvTev(θA, θB, g1, dtau, Q, N, Ne=0, ψA=None, ψB=None,
                      Ansatz=StateAnsatzXZ, offset=0):
    ''''
    Generate overlap circuits with environments.
    '''
    if ψA is None:
        ψA = θA

    if ψB is None:
        ψB = θA

    circuitA = NSiteCircuit_Ansatz_Env(θA, ψA, Q, N, Ne, Ansatz, offset)
    circuitB = NSiteCircuit_Ansatz_Env(θB, ψB, Q, N, Ne, Ansatz, offset)
    time_ev_ops = TimeEvolutionOpLayer(g1, dtau, Q)

    circuit = cirq.Circuit()
    circuit.append(circuitA)
    circuit.append(time_ev_ops)
    circuit.append(cirq.inverse(circuitB))
    return circuit


def OverlapCircuitEnvTevRight(θA, θB, g1, dtau, Q, N, Ne=0, ψA=None, ψB=None,
                      Ansatz=StateAnsatzXZ, offset=0):
    ''''
    Generate overlap circuits with environments.
    '''
    if ψA is None:
        ψA = θA

    if ψB is None:
        ψB = θA

    circuitA = NSiteCircuit_Ansatz_Env_Right(θA, ψA, Q, N, Ne, Ansatz, offset)
    circuitB = NSiteCircuit_Ansatz_Env_Right(θB, ψB, Q, N, Ne, Ansatz, offset)
    time_ev_ops = TimeEvolutionOpLayer(g1, dtau, Q)

    circuit = cirq.Circuit()
    circuit.append(circuitA)
    circuit.append(time_ev_ops)
    circuit.append(cirq.inverse(circuitB))
    return circuit

def AddMeasure(circuit, Qs, name):
    # Add measurements to a circuit in the given locations

    circuit.append(cirq.measure(*Qs, key = name), strategy = cirq.InsertStrategy.NEW_THEN_INLINE)
    return circuit

def SwapTestOps(Qs):
    '''
    Add destructive SWAP test between Qs = [QA, QB]
    '''
    assert len(Qs) == 2, "Need to perform SWAP test across two qubits"
    return [cirq.CNOT(*Qs), cirq.H(Qs[0])]


def Gate_to_Unitary(params, Ansatz):
    return cirq.unitary(Ansatz(params))


if __name__ == '__main__':
    θA = np.random.rand(8)
    θB = np.random.rand(8)
    n = 3

    q = cirq.LineQubit.range(n+1)

    circuit = OverlapCircuitEnv(θA, θB, q, n)
    print(circuit.to_text_diagram(transpose=True))
