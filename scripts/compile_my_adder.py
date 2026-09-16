#!/usr/bin/env python3
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt

from qiskit import transpile, qasm3
from qiskit.quantum_info import get_clifford_gate_names
from qiskit.transpiler.passes import LitinskiTransformation
from qiskit_parser import iter_qiskit_pbc_circuit

# Project directories
WORKSPACE = Path.home() / "bicycle-architecture-compiler"
PBC_DIR = WORKSPACE / "output_pbc"
PBC_DIR.mkdir(parents=True, exist_ok=True)

def compile_pbc(circuit):
    # 1. Strip classical readout measurements to ensure strict unitarity
    circuit.remove_final_measurements()

    # 2. Transpile into fault-tolerant Clifford + T basis
    basis = ["rz", "t", "tdg"] + get_clifford_gate_names()
    tqc = transpile(circuit, basis_gates=basis)

    # 3. Convert to Pauli-Based Computation (PBC) measurements
    lit = LitinskiTransformation(fix_clifford=False)
    pbc = lit(tqc)

    # 4. Stream pure JSON instructions to stdout for bicycle_compiler
    for inst in iter_qiskit_pbc_circuit(pbc):
        print(json.dumps(inst).replace(" ", ""))

    return pbc

if len(sys.argv) < 2:
    print("Error: Please provide the path to your .qasm file.", file=sys.stderr)
    sys.exit(1)

qasm_path = Path(sys.argv[1]).resolve()
if not qasm_path.exists():
    print(f"Error: File not found: {qasm_path}", file=sys.stderr)
    sys.exit(1)

adder_circuit = qasm3.load(str(qasm_path))
num_q = adder_circuit.num_qubits
print(f"Loaded circuit '{qasm_path.name}' with {num_q} allocated logical qubits.", file=sys.stderr)

# Compile to PBC
pbc = compile_pbc(adder_circuit)

# Save PBC circuit diagram ONLY if the circuit is small enough to render safely
if adder_circuit.num_qubits <= 36 and len(adder_circuit.data) < 1000:
    pbc_img_path = PBC_DIR / f"{qasm_path.stem}_pbc.png"
    try:
        fig = pbc.draw("mpl", fold=-1)
        fig.savefig(str(pbc_img_path), bbox_inches='tight', dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not save PBC diagram: {e}", file=sys.stderr)
else:
    print(f"Skipping PBC image export: circuit too large ({adder_circuit.num_qubits} qubits, {len(adder_circuit.data)} ops)", file=sys.stderr)

# python compile_my_adder.py ../4_bit_sklansky_no_gidney.qasm | ../target/release/bicycle_compiler two-gross --measurement-table ../data/table_two-gross | ../target/release/bicycle_numerics 15 two-gross_1e-4