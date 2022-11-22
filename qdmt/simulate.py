import cirq
import numpy as np
import cirq_google as cg
import qsimcirq as qsim
from copy import deepcopy
import os
from qvm import prepare_simulator, transform_circuit

def SimulateCircuitLocalNoiseless(Circuit, Reps):
    # Simulate circuit locally with sampling but no noise

    sim = cirq.Simulator()

    results = sim.run(Circuit, repetitions = Reps)
    return results


def SimulateCircuitLocalNoisy(Circuit, Reps, Noise):
    # Simulate Circuits Locally with sampling and a noise model specified by Noise
    # e.g. Noise = cirq.depolarize(p = 0.01)
    # This is much slower so we are going to use qsimcirq

    Circuit = cg.optimizers.optimized_for_sycamore(Circuit)
    sim = qsim.QSimSimulator({'t':6})
    results = sim.run(Circuit.with_noise(Noise), repetitions = Reps)

    return results


def SimulateCircuitLocalClassicalReadoutError(Circuit, MeasureQubits, Reps, P):
    # Simulate circuits locally with sampling - using a noise model that simulates
    #   classical readout error:
    sim = qsim.QSimSimulator({'t':6})

    noisyCircuit = deepcopy(Circuit)
    noisyCircuit.insert(-1, cirq.bit_flip(p=P).on_each(MeasureQubits))
    #print(noisyCircuit.to_text_diagram(transpose = True))
    results = sim.run(noisyCircuit, repetitions = Reps)

    return results


def SimulateCircuitLocalExact(Circuit, Reps = None, dtype=np.complex64):
    # Simulate circuit locally without sampling

    sim = cirq.Simulator(dtype=dtype)

    results = sim.simulate(Circuit)
    return results

def SimulateCircuitLocalWeber(Circuit, Reps=10000, QubitMap=None):
    '''
    Run a simulation of the circuit using a QVM using Weber's gate noise model.
    By default the circuit is first transformed into the gateset of the device and then the simulation occurs.

    Note by default the simulator expects the measurements to have the key 'measure'
    '''

    circuit = transform_circuit(Circuit, qubit_map=QubitMap)

    sim = prepare_simulator('weber')

    results = sim.run(circuit, repetitions=Reps)

    return results.histogram(key='measure')


def SimulateCircuitLocalRainbow(Circuit, Reps=10000, QubitMap=None):
    '''
    Run a simulation of the circuit using a QVM using Rainbow's gate noise model.
    By default the circuit is first transformed into the gateset of the device and then the simulation occurs.

    Note by default the simulator expects the measurements to have the key 'measure'
    '''

    circuit = transform_circuit(Circuit, qubit_map=QubitMap)

    sim = prepare_simulator('rainbow')

    results = sim.run(circuit, repetitions=Reps)

    return results.histogram(key='measure')


def SimulateCircuitLocalQVM(Circuit, Reps=10000, QubitMap=None, processor_id='weber'):
    '''
    Run a simulation of the circuit using a QVM with the given processor id's
    gate noise model. By default the circuit is first transformed into the
    gateset of the device and then the simulation occurs.

    Note by default the simulator expects the measurements to have the key 'measure'
    '''

    circuit = transform_circuit(Circuit, qubit_map=QubitMap)

    sim = prepare_simulator(processor_id)

    results = sim.run(circuit, repetitions=Reps)

    return results.histogram(key='measure')


