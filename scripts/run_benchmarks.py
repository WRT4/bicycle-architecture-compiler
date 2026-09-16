#!/usr/bin/env python3
import sys
import subprocess
import pandas as pd
from pathlib import Path
import gc
import matplotlib.pyplot as plt

# Add sibling QLDPC-Compilers repository
repo_root = Path.home() / "QLDPC-Compilers"
sys.path.append(str(repo_root))

from Applications.Adders.sklansky_st2 import n_bit_sklansky
from Applications.Adders.brent_kung_st2 import n_bit_brent_kung
from Applications.Adders.ladner_fischer_st2 import n_bit_ladner_fischer
from Applications.Adders.kogge_stone_st2 import n_bit_kogge_stone
from Applications.Adders.han_carlson_st2 import n_bit_han_carlson

from Compiler.frontend.graphs import (
    partition_and_save_graphs,
    remap_circuit_by_partition,
    save_circuit_diagram
)
from qiskit import qasm3

# Directory layout
WORKSPACE = Path.home() / "bicycle-architecture-compiler"
SCRIPTS_DIR = WORKSPACE / "scripts"
DATA_TABLE = WORKSPACE / "data" / "table_two-gross"
COMPILER_BIN = WORKSPACE / "target" / "release" / "bicycle_compiler"
NUMERICS_BIN = WORKSPACE / "target" / "release" / "bicycle_numerics"
OUTPUT_CSV = WORKSPACE / "metrics.csv"

# Target output folders
QASM_DIR = WORKSPACE / "generated_qasm"
GRAPHS_DIR = WORKSPACE / "output_graphs"
PBC_DIR = WORKSPACE / "output_pbc"

# Ensure all target folders exist
for folder in [QASM_DIR, GRAPHS_DIR, PBC_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

def run_pipeline(qasm_file: Path, num_qubits: int) -> dict:
    """Executes compile_my_adder -> bicycle_compiler -> bicycle_numerics."""
    cmd = (
        f"python3 {SCRIPTS_DIR}/compile_my_adder.py {qasm_file} | "
        f"{COMPILER_BIN} two-gross --measurement-table {DATA_TABLE} | "
        f"{NUMERICS_BIN} {num_qubits} two-gross_1e-4"
    )

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running pipeline for {qasm_file.name}:\n{result.stderr}", file=sys.stderr)
        return None

    # Parse final row from CSV output
    lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    if not lines:
        return None
    header = lines[0].split(",")
    final_row = lines[-1].split(",")
    return dict(zip(header, final_row))

def main(n: int):
    adders = {
        f"sklansky_n{n}": n_bit_sklansky(n=n, en_c0=False, use_gidney=False),
        f"brent_kung_n{n}": n_bit_brent_kung(n=n, en_c0=False, use_gidney=False),
        f"ladner_fischer_n{n}": n_bit_ladner_fischer(n=n, en_c0=False, use_gidney=False),
        f"kogge_stone_n{n}": n_bit_kogge_stone(n=n, en_c0=False, use_gidney=False),
        f"han_carlson_n{n}": n_bit_han_carlson(n=n, en_c0=False, use_gidney=False)
    }

    all_results = []

    for adder_name, qc in adders.items():
        print(f"\n{'='*25} Processing {adder_name} {'='*25}")

        # Save baseline unpartitioned circuit diagram
        if qc.num_qubits <= 36:
            save_circuit_diagram(qc, name=f"{adder_name}_00_original", output_dir=str(GRAPHS_DIR))

        # Baseline partition: sequential mapping (0-11 to Mod 0, 12-23 to Mod 1 etc.)
        baseline_part = {q: (q // 12) for q in range(qc.num_qubits)}
        deg_part, metis_part, kahypar_part = partition_and_save_graphs(
            qc, name=adder_name, output_dir=str(GRAPHS_DIR)
        )

        schemes = {
            "original": baseline_part,
            "weighted_degree": deg_part,
            "metis": metis_part,
            "kahypar": kahypar_part
        }

        for scheme_name, part_map in schemes.items():
            remapped_qc = remap_circuit_by_partition(qc, part_map)
            qasm_path = QASM_DIR / f"{adder_name}_{scheme_name}.qasm"

            # 1. Export QASM to generated_qasm/
            with open(qasm_path, "w") as f:
                qasm3.dump(remapped_qc, f)

            # 2. Save remapped circuit diagram to output_graphs/
            if qc.num_qubits <= 36:
                save_circuit_diagram(remapped_qc, name=f"{adder_name}_{scheme_name}", output_dir=str(GRAPHS_DIR))

            # 3. Stream through IBM compiler pipeline (PBC saved to output_pbc/)
            metrics = run_pipeline(qasm_path, remapped_qc.num_qubits)
            if metrics:
                row = {
                    "adder": adder_name,
                    "partition": scheme_name,
                    "allocated_qubits": remapped_qc.num_qubits,
                    "pbc_steps": metrics.get("i"),
                    "measurement_depth": metrics.get("measurement_depth"),
                    "end_time": metrics.get("end_time"),
                    "total_error": metrics.get("total_error"),
                    "failure_prob": metrics.get("total_error"),
                    "automorphisms": metrics.get("automorphisms"),
                    "measurements": metrics.get("measurements")
                }
                all_results.append(row)
                print(f"✓ [{scheme_name.upper():16}] Depth: {row['measurement_depth']:>3} | Cycles: {row['end_time']:>5} | Err: {row['total_error']}")

            plt.close('all')
            gc.collect()
        del qc
        gc.collect()

    # Save consolidated metrics
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n================ BENCHMARKING COMPLETE ================")
    print(f"Metrics written to: {OUTPUT_CSV}")
    print(f"QASM files saved in: {QASM_DIR}")
    print(f"Circuit graphs saved in: {GRAPHS_DIR}")
    print(f"PBC graphs saved in: {PBC_DIR}")

if __name__ == "__main__":
    main(32)