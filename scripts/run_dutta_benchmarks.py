#!/usr/bin/env python3
import sys
import os
import gc
import argparse
import subprocess
import contextlib
import io
import math
from pathlib import Path
import pandas as pd
from qiskit import QuantumCircuit, qasm3

# Add sibling QLDPC-Compilers repository
repo_root = Path.home() / "QLDPC-Compilers"
sys.path.append(str(repo_root))

from Applications.Toffoli.dutta_mapping import (
    create_circuit,
    decompose_toffolis,
)

from Compiler.frontend.graphs import (
    partition_and_save_graphs,
    remap_circuit_by_partition,
    save_circuit_diagram
)
from Compiler.frontend.modular_framework import ModularArchitecture, count_im_instructions

# Default workspace paths
WORKSPACE = Path.home() / "bicycle-architecture-compiler"
SCRIPTS_DIR = WORKSPACE / "scripts"
COMPILER_BIN = WORKSPACE / "target" / "release" / "bicycle_compiler"
NUMERICS_BIN = WORKSPACE / "target" / "release" / "bicycle_numerics"

QASM_DIR = WORKSPACE / "generated_qasm"
GRAPHS_DIR = WORKSPACE / "output_graphs"
PBC_DIR = WORKSPACE / "output_pbc"
SAVED_METRICS_DIR = WORKSPACE / "saved_metrics"

for d in [QASM_DIR, GRAPHS_DIR, PBC_DIR, SAVED_METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run automated benchmarking of QLDPC partitioned Dutta MCT circuits through IBM bicycle tools."
    )
    # Circuit selection: Default sweeps n = 4 to 50
    parser.add_argument(
        "-n",
        "--controls",
        type=int,
        nargs="+",
        default=list(range(4, 51)),
        help="One or more control counts to benchmark (default: 4 to 50).",
    )
    parser.add_argument(
        "-a",
        "--num-ancilla",
        type=int,
        default=2,
        help="Number of ancilla qubits for Dutta mapping (default: 2).",
    )
    parser.add_argument(
        "--decompose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decompose Toffolis into Logical AND / Amy gates before compilation (default: True).",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save circuit diagrams as PNGs (default: False to conserve memory on sweeps).",
    )

    # bicycle_compiler options
    parser.add_argument(
        "--code",
        type=str,
        default="two-gross",
        choices=["gross", "two-gross"],
        help="Bicycle code architecture target.",
    )
    parser.add_argument(
        "--measurement-table",
        type=Path,
        default=WORKSPACE / "data" / "table_two-gross",
        help="Path to Clifford measurement synthesis table.",
    )
    parser.add_argument(
        "--accuracy",
        type=float,
        default=1e-9,
        help="Accuracy of small-angle synthesis in bicycle_compiler.",
    )

    # bicycle_numerics options
    parser.add_argument(
        "--noise-model",
        type=str,
        default="two-gross_1e-4",
        help="Noise model identifier passed to bicycle_numerics (e.g. two-gross_1e-4).",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=11,
        choices=[11, 12],
        help="Logical qubit capacity per module (11 data + 1 ancilla, or 12 full).",
    )

    # Architecture & topology options
    parser.add_argument(
        "--topology",
        type=str,
        default="chain",
        choices=["chain", "all_to_all", "grid_2d"],
        help="Module interconnect topology.",
    )
    parser.add_argument(
        "--factory-period",
        type=int,
        default=2,
        help="Distillation factory periodicity across modules.",
    )

    # Output file
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=SAVED_METRICS_DIR / "dutta_metrics.csv",
        help="Target CSV file for metrics output.",
    )

    return parser.parse_args()


def strip_barriers(circuit: QuantumCircuit) -> QuantumCircuit:
    """Removes barrier instructions so they do not add spurious interaction edges in graph partitioning."""
    clean_qc = circuit.copy_empty_like()
    for instruction in circuit.data:
        if instruction.operation.name != "barrier":
            clean_qc.append(instruction)
    return clean_qc


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


def main():
    args = parse_args()
    
    # Resolve output path
    out_path = args.output if args.output.is_absolute() else (SAVED_METRICS_DIR / args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: load existing results if output file exists
    all_results = []
    completed_configs = set()
    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_path)
            all_results = existing_df.to_dict("records")
            completed_configs = set(zip(existing_df["adder"], existing_df["partition"]))
            print(f"Loaded {len(existing_df)} existing records from {out_path.name}. Resuming...")
        except Exception as e:
            print(f"Could not read existing {out_path.name} ({e}). Starting fresh.")

    total_sweeps = len(args.controls)
    print(f"\nBenchmarking Dutta MCT circuits: n = {min(args.controls)} to {max(args.controls)} ({total_sweeps} values)")
    print(f"Settings: capacity={args.capacity}, ancilla={args.num_ancilla}, topology={args.topology}, code={args.code}")

    for idx, n_val in enumerate(args.controls, 1):
        circuit_name = f"dutta_n{n_val}"
        print(f"\n[{idx}/{total_sweeps}] {'='*25} Processing {circuit_name} {'='*25}")

        # 1. Generate Dutta MCT circuit (suppress stdout prints from dutta_mapping.py)
        with contextlib.redirect_stdout(io.StringIO()):
            qc, _ = create_circuit(control_size=n_val, num_ancilla=args.num_ancilla)

        # Decompose into Clifford+T if requested
        if args.decompose:
            qc = decompose_toffolis(qc)

        # Strip barriers before hypergraph construction
        qc = strip_barriers(qc)

        num_qubits = qc.num_qubits
        num_modules = max(2, math.ceil(num_qubits / args.capacity))
        print(f"Circuit {circuit_name}: {num_qubits} qubits, {len(qc.data)} gates -> {num_modules} Gross modules")

        # 2. Partitioning Schemes
        baseline_part = {q: (q // args.capacity) for q in range(num_qubits)}
        deg_part, metis_part, kahypar_part, cluster_part = partition_and_save_graphs(
            qc, name=circuit_name, output_dir=str(GRAPHS_DIR), save=args.plot
        )

        # 3. Instantiate Architecture Framework for analytical IM traffic
        arch = ModularArchitecture(
            num_modules=num_modules,
            module_capacity=args.capacity,
            topology=args.topology,
            factory_period=args.factory_period,
        )

        schemes = {
            "original": baseline_part,
            "weighted_degree": deg_part,
            "metis": metis_part,
            "kahypar": kahypar_part,
            "clustering": cluster_part,
        }

        # 4. Compile and benchmark each partition scheme
        for scheme_name, part_map in schemes.items():
            config_key = (circuit_name, scheme_name)
            if config_key in completed_configs:
                print(f"  ↪ Skipping already completed: {circuit_name} [{scheme_name}]")
                continue

            remapped_qc = remap_circuit_by_partition(qc, part_map, module_capacity=args.capacity)
            qasm_path = QASM_DIR / f"{circuit_name}_{scheme_name}.qasm"
            
            # Analytical hardware traffic
            arch.allocate_qubits(part_map)
            im_metrics = count_im_instructions(qc, arch)

            with open(qasm_path, "w") as f:
                qasm3.dump(remapped_qc, f)

            # Diagram plot guard
            if args.plot:
                if remapped_qc.num_qubits <= 36 and len(remapped_qc.data) < 1000:
                    save_circuit_diagram(
                        remapped_qc,
                        name=f"{circuit_name}_{scheme_name}",
                        output_dir=str(GRAPHS_DIR),
                    )
                else:
                    print(f"  Skipping diagram for {circuit_name} ({remapped_qc.num_qubits}q): exceeds memory limit.")

            # Run compiler -> numerics
            metrics = run_pipeline(qasm_path, remapped_qc.num_qubits, args)
            if metrics:
                row = {
                    "circuit": circuit_name,
                    "partition": scheme_name,
                    "allocated_qubits": remapped_qc.num_qubits,
                    "code": args.code,
                    "noise_model": args.noise_model,
                    "pbc_steps": metrics.get("i"),
                    "measurement_depth": metrics.get("measurement_depth"),
                    "end_time": metrics.get("end_time"),
                    "total_error": metrics.get("total_error"),
                    "automorphisms": metrics.get("automorphisms"),
                    "measurements": metrics.get("measurements"),
                    "topology": args.topology,
                    "im_gates": im_metrics["inter_module_gates"],
                    "im_hops": im_metrics["inter_module_hops"],
                    "factory_period": args.factory_period,
                    "factory_im_hops": im_metrics["factory_delivery_hops"],
                }
                all_results.append(row)
                completed_configs.add(config_key)
                print(
                    f"  ✓ [{scheme_name.upper():16}] "
                    f"Depth: {row['measurement_depth']:>4} | "
                    f"Cycles: {row['end_time']:>7} | "
                    f"IM Gates: {im_metrics['inter_module_gates']:>3} | "
                    f"Err: {row['total_error']}"
                )

            del remapped_qc
            gc.collect()

        # Flush intermediate results after each n value so progress is preserved
        pd.DataFrame(all_results).to_csv(out_path, index=False)

        del qc
        gc.collect()

    print(f"\n{'='*30} Benchmark Complete {'='*30}")
    print(f"Total configurations recorded: {len(all_results)}")
    print(f"Metrics successfully written to: {out_path}")


if __name__ == "__main__":
    main()