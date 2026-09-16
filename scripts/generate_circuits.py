#!/usr/bin/env python3
import sys
from pathlib import Path

repo_root = Path.home() / "QLDPC-Compilers"
sys.path.append(str(repo_root))

from Applications.Adders.sklansky_st2 import n_bit_sklansky
from Compiler.frontend.graphs import (
    partition_and_save_graphs,
    remap_circuit_by_partition,
    save_circuit_diagram
)
from qiskit import qasm3

def main():
    # 1. Generate 4-bit Sklansky adder circuit (unitary uncomputation)
    sk = n_bit_sklansky(n=4, en_c0=False, use_gidney=False)
    save_circuit_diagram(sk, name="sklansky_n4_00_original")

    # 2. Partition across 12-qubit Bivariate Bicycle modules
    deg_part, metis_part, kahypar_part = partition_and_save_graphs(sk, name="sklansky_n4")

    partitions = {
        "weighted_degree": deg_part,
        "metis": metis_part,
        "kahypar": kahypar_part
    }

    # 3. Export partitioned circuits and diagrams
    for method_name, part_map in partitions.items():
        remapped_qc = remap_circuit_by_partition(sk, part_map)
        
        qasm_path = f"sklansky_n4_{method_name}.qasm"
        with open(qasm_path, "w") as f:
            qasm3.dump(remapped_qc, f)
        print(f"Exported QASM3 circuit to: {qasm_path}")

        save_circuit_diagram(remapped_qc, name=f"sklansky_n4_{method_name}")

if __name__ == "__main__":
    main()