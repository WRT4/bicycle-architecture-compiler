#!/usr/bin/env python3
import sys
import json

from qiskit import transpile, qasm3
from qiskit.quantum_info import get_clifford_gate_names
from qiskit.transpiler.passes import LitinskiTransformation

from qiskit_parser import iter_qiskit_pbc_circuit

def compile_pbc(circuit):
    # 1. Strip classical readout measurements to make it 100% unitary
    circuit.remove_final_measurements()
    
    # 2. Break the circuit down into the fault-tolerant gate set
    basis = ["rz", "t", "tdg"] + get_clifford_gate_names()
    tqc = transpile(circuit, basis_gates=basis)

    # 3. Convert to Pauli-Based Computation (PBC) measurements
    lit = LitinskiTransformation(fix_clifford=False)
    pbc = lit(tqc)

    # 4. Stream to the Rust compiler
    for inst in iter_qiskit_pbc_circuit(pbc):
        print(json.dumps(inst).replace(" ", ""))
    
    return pbc

if len(sys.argv) < 2:
    print("Error: Please provide the path to your .qasm file.", file=sys.stderr)
    sys.exit(1)

qasm_file = sys.argv[1]
adder_circuit = qasm3.load(qasm_file)

# Tell the user exactly how many qubits this circuit uses!
num_q = adder_circuit.num_qubits
print(f"Loaded circuit with exactly {num_q} logical qubits.", file=sys.stderr)
print(f"--> Ensure you use {num_q} in the bicycle_numerics command!", file=sys.stderr)

pbc = compile_pbc(adder_circuit)
pbc.draw("mpl", filename="sk_qasm.png")


# python compile_my_adder.py ../4_bit_sklansky_no_gidney.qasm | ../target/release/bicycle_compiler two-gross --measurement-table ../data/table_two-gross | ../target/release/bicycle_numerics 15 two-gross_1e-4
