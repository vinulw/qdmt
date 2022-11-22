import cirq
import cirq_google
import qsimcirq
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from circuits import OverlapCircuitEnvTev, AddMeasure

def plot_errors(processor_id):
    cal = cirq_google.engine.load_median_device_calibration(processor_id)
    noise_props = cirq_google.noise_properties_from_calibration(cal)

    # Plot single qubit pauli errors
    fig, ax = plt.subplots(figsize=(10, 10))
    gate = cirq.PhasedXZGate
    measures = {
        op_id.qubits: pauli_error
        for op_id, pauli_error in noise_props.gate_pauli_errors.items()
        if op_id.gate_type == gate
    }
    ax.set_title(f"{gate.__name__} Single Qubit Pauli error")
    _ = cirq.Heatmap(measures).plot(ax)

    # Two qubit Pauli decoherence

    two_qubit_gates = noise_props.two_qubit_gates()
    print(f"Two qubit error data: gate_pauli_errors")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes = iter(axes)
    for gate in two_qubit_gates:
        measures = {
            op_id.qubits: pauli_error
            for op_id, pauli_error in noise_props.gate_pauli_errors.items()
            if op_id.gate_type == gate
        }
        if measures:
            ax = next(axes)
            ax.set_title(f"{gate.__name__} Two Qubit Pauli error")
            _ = cirq.TwoQubitInteractionHeatmap(measures).plot(ax)

    # Plot 2 qubit Fsim error
    two_qubit_gates = noise_props.two_qubit_gates()
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes = iter(axes)
    for gate in two_qubit_gates:
        measures = {
            op_id.qubits: fsim_refit_gate
            for op_id, fsim_refit_gate in noise_props.fsim_errors.items()
            if op_id.gate_type == gate
        }
        if measures:
            ax = next(axes)
            # Norm the Fsim refit gate parameters as an approximate of how good a qubit is.
            measures = {
                qubits: np.linalg.norm([fsim_refit_gate.theta, fsim_refit_gate.phi])
                for qubits, fsim_refit_gate in measures.items()
            }
            ax.set_title(f"{gate.__name__} FSim Pauli error")
            _ = cirq.TwoQubitInteractionHeatmap(measures).plot(ax)

    # Plot readout error

    print(f"One qubit error data: readout_errors")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for i, ax, title in zip(
        range(2), axes.flat, ["True |0> readout measured as |1>", "True |1> readout measured as |0>"]
    ):
        measures = {
            qubit: readout_error[i] for qubit, readout_error in noise_props.readout_errors.items()
        }
        ax.set_title(title)
        _ = cirq.Heatmap(measures).plot(ax, vmax=0.4, vmin=0)
    plt.show()

def transform_circuit(circuit, qubit_map=None, gateset=None):
    '''
    Prepare the circuit to run on a QVM. To do so requires two changes.

    1) Change the gateset to one that is present on the device.
    2) Modify line qubits from the circuit to a selection of grid qubit present
       on the device

    By default no changes are made to the qubits and the gateset of the Weber
    device is chosen (SqrtIswap)
    '''
    if gateset is None:
        gateset = cirq.SqrtIswapTargetGateset()

    translated_circuit = cirq.optimize_for_target_gateset(circuit, context=cirq.TransformerContext(deep=True),
                                                          gateset=gateset)

    if qubit_map is not None:
        translated_circuit = translated_circuit.transform_qubits(lambda q: qubit_map[q])

    return translated_circuit

def prepare_simulator(processor_id='weber'):
    # Load median device noise data for given processor
    cal = cirq_google.engine.load_median_device_calibration(processor_id)
    # Create noise properties object from calibration measurements
    noise_props = cirq_google.noise_properties_from_calibration(cal)
    # Create a noise model for simulation from noise properties
    noise_model = cirq_google.NoiseModelFromGoogleNoiseProperties(noise_props)
    # Prepare qsim simulator using the noise model
    sim = qsimcirq.QSimSimulator(noise=noise_model)

    return sim

def prepare_processor_and_engine(processor_id='weber'):
    # Load median device noise data for given processor
    cal = cirq_google.engine.load_median_device_calibration(processor_id)
    # Create noise properties object from calibration measurements
    noise_props = cirq_google.noise_properties_from_calibration(cal)
    # Create a noise model for simulation from noise properties
    noise_model = cirq_google.NoiseModelFromGoogleNoiseProperties(noise_props)
    # Prepare qsim simulator using the noise model
    sim = qsimcirq.QSimSimulator(noise=noise_model)

	# Package the simulator and device in an Engine.
    # The device object
    device = cirq_google.engine.create_device_from_processor_id(processor_id)
    # The simulated processor object
    sim_processor = cirq_google.engine.SimulatedLocalProcessor(
        processor_id=processor_id, sampler=sim, device=device, calibrations={cal.timestamp // 1000: cal}
    )
    # The virtual engine
    sim_engine = cirq_google.engine.SimulatedLocalEngine([sim_processor])

    return sim_processor, sim_engine

def main():
    from circuits import OverlapCircuitEnv
    from simulate import SimulateCircuitLocalExact, SimulateCircuitLocalNoiseless, SimulateCircuitLocalWeber
    from tqdm import tqdm
    n_range = np.arange(2, 10)
    shots = 10000
    print(n_range)

    device_qubit_list = [
            cirq.GridQubit(1, 4),
            cirq.GridQubit(2, 4),
            cirq.GridQubit(3, 4),
            cirq.GridQubit(4, 4),
            cirq.GridQubit(5, 4),
            cirq.GridQubit(6, 4),
            cirq.GridQubit(7, 4),
            cirq.GridQubit(7, 5),
            cirq.GridQubit(6, 5),
            cirq.GridQubit(5, 5),
            cirq.GridQubit(4, 5),
            ]

    alpha = 0.5
    θA = np.random.rand(8)
    θB = θA + alpha*np.random.rand(8)

    probs_exact = []
    probs_noiseless = []
    probs_weber = []
    for n in tqdm(n_range):
        Q = cirq.LineQubit.range(n+1)
        circuitn = OverlapCircuitEnv(θA, θB, Q, n)
        res = SimulateCircuitLocalExact(circuitn)
        prob0 = np.abs(res.final_state_vector[0], dtype=float)**2
        probs_exact.append(prob0)

        circuitn = AddMeasure(circuitn, Q, 'measure')
        res = SimulateCircuitLocalNoiseless(circuitn, shots).histogram(key='measure')
        prob0 = res[0] / sum(res.values())
        probs_noiseless.append(prob0)

        qubit_map = dict(zip(Q, device_qubit_list[:n+1]))
        res = SimulateCircuitLocalWeber(circuitn, shots, qubit_map)
        prob0 = res[0] / sum(res.values())
        probs_weber.append(prob0)

    probs_exact = np.array(probs_exact)
    overlaps_exact = probs_exact[1:]/probs_exact[:-1]
    probs_noiseless = np.array(probs_noiseless)
    overlaps_noiseless = probs_noiseless[1:]/probs_noiseless[:-1]
    probs_weber = np.array(probs_weber)
    overlaps_weber = probs_weber[1:]/probs_weber[:-1]

    plt.plot(n_range[:-1], overlaps_exact, label='exact')
    plt.plot(n_range[:-1], overlaps_noiseless, label='noiseless')
    plt.plot(n_range[:-1], overlaps_weber, label='weber')
    plt.legend()
    plt.show()


if __name__=="__main__":
    main()
