#!/usr/bin/env python3
import sys
import os
import gc
import argparse
import subprocess
import pandas as pd
from pathlib import Path

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

# Default workspace paths
WORKSPACE = Path.home() / "bicycle-architecture-compiler"
SCRIPTS_DIR = WORKSPACE / "scripts"
COMPILER_BIN = WORKSPACE / "target" / "release" / "bicycle_compiler"
NUMERICS_BIN = WORKSPACE / "target" / "release" / "bicycle_numerics"

QASM_DIR = WORKSPACE / "generated_qasm"
GRAPHS_DIR = WORKSPACE / "output_graphs"
PBC_DIR = WORKSPACE / "output_pbc"

for d in [QASM_DIR, GRAPHS_DIR, PBC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run automated benchmarking of QLDPC partitioned adders through IBM bicycle tools."
    )
    # Circuit selection
    parser.add_argument("-n", "--bit-widths", type=int, nargs="+", default=[8],
                        help="One or more adder bit-widths to benchmark (e.g. -n 8 16 32).")
    parser.add_argument("--adders", type=str, nargs="+", 
                        default=["sklansky", "brent_kung", "ladner_fischer", "kogge_stone", "han_carlson"],
                        help="Adder families to benchmark.")

    # bicycle_compiler options
    parser.add_argument("--code", type=str, default="two-gross", choices=["gross", "two-gross"],
                        help="Bicycle code architecture target.")
    parser.add_argument("--measurement-table", type=Path, 
                        default=WORKSPACE / "data" / "table_two-gross",
                        help="Path to Clifford measurement synthesis table.")
    parser.add_argument("-a", "--accuracy", type=float, default=1e-9,
                        help="Accuracy of small-angle synthesis in bicycle_compiler.")

    # bicycle_numerics options
    parser.add_argument("--noise-model", type=str, default="two-gross_1e-4",
                        help="Noise model identifier passed to bicycle_numerics (e.g. two-gross_1e-4).")

    # Output file
    parser.add_argument("-o", "--output", type=Path, default=WORKSPACE / "metrics.csv",
                        help="Target CSV file for metrics output.")
    
    return parser.parse_args()

def run_pipeline(qasm_file: Path, num_qubits: int, args) -> dict:
    """Streams compile_my_adder -> bicycle_compiler -> bicycle_numerics with dynamic CLI args."""
    cmd = (
        f"python3 {SCRIPTS_DIR}/compile_my_adder.py {qasm_file} | "
        f"{COMPILER_BIN} {args.code} --measurement-table {args.measurement_table} --accuracy {args.accuracy} | "
        f"{NUMERICS_BIN} {num_qubits} {args.noise_model}"
    )

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Pipeline error for {qasm_file.name}:\n{result.stderr}", file=sys.stderr)
        return None

    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    if not lines:
        return None
    header = lines[0].split(",")
    final_row = lines[-1].split(",")
    return dict(zip(header, final_row))

def get_adder_constructor(family: str):
    lookup = {
        "sklansky": n_bit_sklansky,
        "brent_kung": n_bit_brent_kung,
        "ladner_fischer": n_bit_ladner_fischer,
        "kogge_stone": n_bit_kogge_stone,
        "han_carlson": n_bit_han_carlson,
    }
    return lookup.get(family)

def main():
    args = parse_args()
    all_results = []

    for n_val in args.bit_widths:
        for fam in args.adders:
            constructor = get_adder_constructor(fam)
            if not constructor:
                print(f"Skipping unknown adder family: {fam}")
                continue

            adder_name = f"{fam}_n{n_val}"
            print(f"\n{'='*25} Processing {adder_name} {'='*25}")
            qc = constructor(n=n_val, en_c0=False, use_gidney=False)

            # Sequential multi-module baseline (q // 12)
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

                with open(qasm_path, "w") as f:
                    qasm3.dump(remapped_qc, f)

                # Skip drawing circuit diagrams for high-depth circuits to prevent RAM exhaust
                if qc.num_qubits <= 36 and len(qc.data) < 1000:
                    save_circuit_diagram(remapped_qc, name=f"{adder_name}_{scheme_name}", output_dir=str(GRAPHS_DIR))

                metrics = run_pipeline(qasm_path, remapped_qc.num_qubits, args)
                if metrics:
                    row = {
                        "adder": adder_name,
                        "partition": scheme_name,
                        "allocated_qubits": remapped_qc.num_qubits,
                        "code": args.code,
                        "noise_model": args.noise_model,
                        "pbc_steps": metrics.get("i"),
                        "measurement_depth": metrics.get("measurement_depth"),
                        "end_time": metrics.get("end_time"),
                        "total_error": metrics.get("total_error"),
                        "failure_prob": metrics.get("total_error"),
                        "automorphisms": metrics.get("automorphisms"),
                        "measurements": metrics.get("measurements")
                    }
                    all_results.append(row)
                    print(f"✓ [{scheme_name.upper():16}] Depth: {row['measurement_depth']:>4} | Cycles: {row['end_time']:>7} | Err: {row['total_error']}")

                del remapped_qc
                gc.collect()

            del qc
            gc.collect()

    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"\nMetrics written to: {args.output}")

if __name__ == "__main__":
    main()

# python3 run_benchmarks.py -n 4 8 16 32 --noise-model two-gross_1e-3