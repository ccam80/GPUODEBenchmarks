#!/usr/bin/env python3
"""Compare CuBIE Euler with Myokit's generated CUDA Euler on Fabbri-Linder.

Run the orchestrator with the Myokit-CUDA environment.  CuBIE is invoked in
its own environment through ``--cubie-python`` so neither benchmark needs a
combined environment.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import timeit
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MYOKIT_DIR = REPO_ROOT / "GPU_ODE_MYOKIT_CUDA"
sys.path.insert(0, str(MYOKIT_DIR))
sys.path.insert(0, str(REPO_ROOT / "runner_scripts"))

from bench_key import dataset_key  # noqa: E402
from myokit_cuda import MyokitCudaModel  # noqa: E402


DEFAULT_DT = 1e-5
DEFAULT_STEPS = 1000
CELLML_NAMESPACE = "http://www.cellml.org/cellml/1.0#"
CELLML = "{{{0}}}".format(CELLML_NAMESPACE)
FABBRI_REPAIRS = (
    "Ca_buffering.kb_CM public_interface set to out",
    "Ca_buffering.kf_CM public_interface set to out",
    "Ca_buffering.fCMi public_interface set to out",
    "duplicate ATPi/cAMP connections merged into cAMP-to-ATPi connection",
)


def positive_integer(value):
    """Parse a strictly positive integer for command-line options."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        )
    return parsed


def normalized_fabbri_cellml(source, destination):
    """Write the metadata-only Myokit compatibility normalization.

    The Fabbri-Linder fixture contains three connected outputs without
    ``public_interface='out'`` and represents the two mappings between cAMP
    and ATPi as oppositely ordered duplicate connections. Myokit's strict
    CellML importer rejects both constructs. This function asserts that exact
    source structure, adjusts only the interface/connection metadata, and
    leaves the original fixture untouched for CuBIE.
    """
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    source_text = source.read_text(encoding="utf-8")
    tree = ElementTree.parse(source)
    root = tree.getroot()
    if root.tag != CELLML + "model":
        raise ValueError(
            "expected a CellML 1.0 model, got {0}".format(root.tag)
        )

    components = {
        component.attrib.get("name"): component
        for component in root.findall(CELLML + "component")
    }
    buffering = components.get("Ca_buffering")
    if buffering is None:
        raise ValueError("expected component Ca_buffering")
    variables = {
        variable.attrib.get("name"): variable
        for variable in buffering.findall(CELLML + "variable")
    }
    for name in ("kb_CM", "kf_CM", "fCMi"):
        variable = variables.get(name)
        if variable is None:
            raise ValueError(
                "expected Ca_buffering.{0}".format(name)
            )
        if "public_interface" in variable.attrib:
            raise ValueError(
                "expected Ca_buffering.{0} without public_interface"
                .format(name)
            )

    pair_connections = []
    for connection in root.findall(CELLML + "connection"):
        mapping = connection.find(CELLML + "map_components")
        if mapping is None:
            continue
        pair = (
            mapping.attrib.get("component_1"),
            mapping.attrib.get("component_2"),
        )
        if set(pair) == {"ATPi", "cAMP"}:
            variable_pairs = [
                (
                    item.attrib.get("variable_1"),
                    item.attrib.get("variable_2"),
                )
                for item in connection.findall(
                    CELLML + "map_variables"
                )
            ]
            pair_connections.append((connection, pair, variable_pairs))

    expected_connections = {
        ("cAMP", "ATPi"): [("ATPi", "ATPi")],
        ("ATPi", "cAMP"): [("cAMP", "cAMP")],
    }
    actual_connections = {
        pair: variable_pairs
        for _, pair, variable_pairs in pair_connections
    }
    if (
        len(pair_connections) != 2
        or actual_connections != expected_connections
    ):
        raise ValueError(
            "unexpected ATPi/cAMP connection structure: {0}".format(
                actual_connections
            )
        )

    exact_variable_fragments = {
        "fCMi": (
            'initial_value="0.217311" name="fCMi" '
            'units="dimensionless"/>'
        ),
        "kf_CM": (
            'initial_value="1.642e6" name="kf_CM" '
            'units="per_millimolar_second"/>'
        ),
        "kb_CM": (
            'initial_value="542" name="kb_CM" units="per_second"/>'
        ),
    }
    normalized = source_text
    for name, fragment in exact_variable_fragments.items():
        if normalized.count(fragment) != 1:
            raise ValueError(
                "expected one exact source fragment for "
                "Ca_buffering.{0}".format(name)
            )
        normalized = normalized.replace(
            fragment,
            fragment.replace(
                " units=", ' public_interface="out" units='
            ),
            1,
        )

    forward_connection = (
        '    <connection>\n'
        '        <map_components component_1="cAMP" '
        'component_2="ATPi"/>\n'
        '        <map_variables variable_1="ATPi" '
        'variable_2="ATPi"/>\n'
        '    </connection>'
    )
    merged_connection = (
        '    <connection>\n'
        '        <map_components component_1="cAMP" '
        'component_2="ATPi"/>\n'
        '        <map_variables variable_1="ATPi" '
        'variable_2="ATPi"/>\n'
        '        <map_variables variable_1="cAMP" '
        'variable_2="cAMP"/>\n'
        '    </connection>'
    )
    reverse_connection = (
        '    <connection>\n'
        '        <map_components component_1="ATPi" '
        'component_2="cAMP"/>\n'
        '        <map_variables variable_1="cAMP" '
        'variable_2="cAMP"/>\n'
        '    </connection>\n'
    )
    if normalized.count(forward_connection) != 1:
        raise ValueError(
            "expected one exact cAMP-to-ATPi connection block"
        )
    if normalized.count(reverse_connection) != 1:
        raise ValueError(
            "expected one exact ATPi-to-cAMP connection block"
        )
    normalized = normalized.replace(
        forward_connection, merged_connection, 1
    )
    normalized = normalized.replace(reverse_connection, "", 1)
    destination.write_text(
        normalized,
        encoding="utf-8",
    )
    return FABBRI_REPAIRS


def canonical_state_name(name):
    """Map Myokit ``component.variable`` names to CuBIE state names."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name.replace(".", "_"))


def timed_myokit_solve(model, trajectories, dt, steps, repeats):
    """Return synchronized minimum time and final states for Myokit-CUDA."""
    initial_states = model.initial_states(trajectories)
    diffusion = np.zeros(trajectories, dtype=np.float32)

    def run():
        return model.solve(
            dt=dt,
            step_count=steps,
            initial_states=initial_states,
            diffusion_values=diffusion,
        )

    finals = run()
    samples = timeit.repeat(
        run,
        setup="gc.enable()",
        repeat=repeats,
        number=1,
    )
    return min(samples) * 1000.0, finals, samples


def cubie_worker(arguments):
    """Run CuBIE inside the interpreter selected by the orchestrator."""
    import cubie
    from cubie.cuda_simsafe import cupy

    cellml = str(Path(arguments.cellml).resolve())
    system = cubie.load_cellml_model(
        cellml,
        precision=np.float32,
        fix_singularities=False,
    )
    state_info = system.get_states_info()
    state_names = tuple(item["name"] for item in state_info)
    initial_values = {
        item["name"]: np.full(
            arguments.trajectories,
            item["value"],
            dtype=np.float32,
        )
        for item in state_info
    }
    duration = arguments.dt * arguments.steps
    solver = cubie.Solver(
        system,
        algorithm="euler",
        dt=arguments.dt,
        save_every=duration,
        step_controller="fixed",
        output_types=["state"],
        time_logging_level=None,
    )
    initials_array, parameter_array = solver.build_grid(
        initial_values=initial_values,
        parameters={},
    )

    def run():
        result = solver.solve(
            initial_values=initials_array,
            parameters=parameter_array,
            blocksize=arguments.block_size,
            duration=duration,
        )
        cupy.cuda.get_current_stream().synchronize()
        values = result.as_numpy["time_domain_array"]
        return np.asarray(values[-1, :, :].T, dtype=np.float32)

    finals = run()
    samples = []
    for _ in range(arguments.repeats):
        start = timeit.default_timer()
        finals = run()
        samples.append(timeit.default_timer() - start)

    np.savez_compressed(
        arguments.output,
        state_names=np.asarray(state_names),
        finals=finals,
        samples_seconds=np.asarray(samples, dtype=np.float64),
    )
    return 0


def run_cubie_subprocess(arguments, output, trajectories):
    """Invoke the hidden worker command in the selected CuBIE environment."""
    command = [
        str(Path(arguments.cubie_python).resolve()),
        str(Path(__file__).resolve()),
        "cubie-worker",
        "--cellml",
        str(Path(arguments.cellml).resolve()),
        "--output",
        str(output),
        "--trajectories",
        str(trajectories),
        "--dt",
        repr(arguments.dt),
        "--steps",
        str(arguments.steps),
        "--repeats",
        str(arguments.repeats),
        "--block-size",
        str(arguments.cubie_block_size),
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def mapped_cubie_states(myokit_names, cubie_names, cubie_finals):
    """Reorder CuBIE final states into Myokit's generated state order."""
    cubie_by_name = {}
    for index, name in enumerate(cubie_names):
        canonical = canonical_state_name(str(name))
        if canonical in cubie_by_name:
            raise ValueError(
                "duplicate canonical CuBIE state name: {0}".format(
                    canonical
                )
            )
        cubie_by_name[canonical] = index

    expected = [canonical_state_name(name) for name in myokit_names]
    missing = [name for name in expected if name not in cubie_by_name]
    extra = sorted(set(cubie_by_name) - set(expected))
    if missing or extra:
        raise ValueError(
            "state-name mismatch; missing in CuBIE: {0}; extra in CuBIE: "
            "{1}".format(missing, extra)
        )
    indices = [cubie_by_name[name] for name in expected]
    return cubie_finals[:, indices], expected


def accuracy_rows(names, myokit_finals, cubie_finals):
    """Calculate global and per-state float64 difference statistics."""
    difference = (
        np.asarray(myokit_finals, dtype=np.float64)
        - np.asarray(cubie_finals, dtype=np.float64)
    )
    absolute = np.abs(difference)
    denominator = np.maximum(np.abs(cubie_finals), 1e-30)
    relative = absolute / denominator
    rows = []
    for index, name in enumerate(names):
        state_difference = difference[:, index]
        state_absolute = absolute[:, index]
        state_relative = relative[:, index]
        rows.append(
            {
                "state": name,
                "maximum_absolute_error": float(np.max(state_absolute)),
                "mean_absolute_error": float(np.mean(state_absolute)),
                "root_mean_square_error": float(
                    np.sqrt(np.mean(state_difference ** 2))
                ),
                "maximum_relative_error": float(
                    np.max(state_relative)
                ),
            }
        )
    summary = {
        "maximum_absolute_error": float(np.max(absolute)),
        "mean_absolute_error": float(np.mean(absolute)),
        "root_mean_square_error": float(
            np.sqrt(np.mean(difference ** 2))
        ),
        "maximum_relative_error": float(np.max(relative)),
    }
    return rows, summary


def write_outputs(
    output_dir,
    arguments,
    state_names,
    myokit_finals,
    cubie_finals,
    myokit_ms,
    cubie_ms,
    myokit_samples,
    cubie_samples,
    cellml_repairs,
    trajectories,
):
    """Write timing samples, accuracy CSV, JSON, and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, accuracy = accuracy_rows(
        state_names,
        myokit_finals,
        cubie_finals,
    )
    with (output_dir / "accuracy_by_state.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "timings.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("implementation", "repeat", "milliseconds"))
        for implementation, samples in (
            ("myokit_cuda", myokit_samples),
            ("cubie", cubie_samples),
        ):
            for repeat, sample in enumerate(samples, start=1):
                writer.writerow(
                    (implementation, repeat, float(sample) * 1000.0)
                )

    allclose = bool(
        np.allclose(
            myokit_finals,
            cubie_finals,
            rtol=arguments.rtol,
            atol=arguments.atol,
        )
    )
    report = {
        "cellml": str(Path(arguments.cellml).resolve()),
        "dataset_key": dataset_key(),
        "precision": "float32",
        "algorithm": "forward_euler",
        "dt": arguments.dt,
        "steps": arguments.steps,
        "duration": arguments.dt * arguments.steps,
        "trajectories": trajectories,
        "state_count": len(state_names),
        "state_names": list(state_names),
        "repeats": arguments.repeats,
        "timing_statistic": "minimum",
        "block_size": {
            "myokit_cuda": arguments.myokit_block_size,
            "cubie": arguments.cubie_block_size,
        },
        "warmup_excluded": True,
        "synchronized": True,
        "myokit_cellml_repairs": list(cellml_repairs),
        "myokit_generated_source_adjustments": [
            "removed exact unused #include <float.h> for NVRTC",
            "appended ensemble launch kernel",
        ],
        "timing_milliseconds": {
            "myokit_cuda_minimum": myokit_ms,
            "cubie_minimum": cubie_ms,
            "cubie_over_myokit_cuda": cubie_ms / myokit_ms,
            "myokit_cuda_over_cubie": myokit_ms / cubie_ms,
        },
        "accuracy": accuracy,
        "allclose": {
            "rtol": arguments.rtol,
            "atol": arguments.atol,
            "result": allclose,
        },
    }
    with (output_dir / "comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    markdown = [
        "# CuBIE vs Myokit-CUDA: Fabbri-Linder\n\n",
        "- CellML: `{0}`\n".format(report["cellml"]),
        "- Precision: float32\n",
        "- Algorithm: forward Euler in both implementations\n",
        "- Step: `{0:g}` s x `{1}` (`{2:g}` s total)\n".format(
            arguments.dt,
            arguments.steps,
            report["duration"],
        ),
        "- Trajectories: `{0}`\n".format(trajectories),
        "- States: `{0}` (mapped by `component.variable` to "
        "`component_variable`)\n".format(len(state_names)),
        "- Myokit-only CellML metadata repairs: `{0}`\n".format(
            len(cellml_repairs)
        ),
        "- CUDA block sizes: Myokit-CUDA `{0}`, CuBIE `{1}`\n".format(
            arguments.myokit_block_size,
            arguments.cubie_block_size,
        ),
        "- Timing: minimum of `{0}` synchronized repeats; compilation and "
        "one warmup solve excluded.\n\n".format(arguments.repeats),
        "## Timing\n\n",
        "| Implementation | Minimum (ms) |\n",
        "|---|---:|\n",
        "| Myokit-CUDA | {0:.6f} |\n".format(myokit_ms),
        "| CuBIE | {0:.6f} |\n".format(cubie_ms),
        "\nCuBIE / Myokit-CUDA: `{0:.6f}`.\n\n".format(
            cubie_ms / myokit_ms
        ),
        "CuBIE speedup: `{0:.6f}x`.\n\n".format(
            myokit_ms / cubie_ms
        ),
        "## Accuracy\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        "| Maximum absolute error | {0:.9e} |\n".format(
            accuracy["maximum_absolute_error"]
        ),
        "| Mean absolute error | {0:.9e} |\n".format(
            accuracy["mean_absolute_error"]
        ),
        "| RMS error | {0:.9e} |\n".format(
            accuracy["root_mean_square_error"]
        ),
        "| Maximum relative error | {0:.9e} |\n".format(
            accuracy["maximum_relative_error"]
        ),
        "\n`numpy.allclose(rtol={0:g}, atol={1:g})`: `{2}`.\n".format(
            arguments.rtol,
            arguments.atol,
            allclose,
        ),
    ]
    (output_dir / "comparison.md").write_text(
        "".join(markdown),
        encoding="utf-8",
    )
    return report


def write_scaling_summary(output_dir, reports):
    """Write aggregate CSV and Markdown tables for every trajectory count."""
    fields = (
        "trajectories",
        "myokit_cuda_minimum_ms",
        "cubie_minimum_ms",
        "cubie_over_myokit_cuda",
        "myokit_cuda_over_cubie",
        "maximum_absolute_error",
        "root_mean_square_error",
        "allclose",
    )
    rows = []
    for report in reports:
        timing = report["timing_milliseconds"]
        accuracy = report["accuracy"]
        rows.append(
            {
                "trajectories": report["trajectories"],
                "myokit_cuda_minimum_ms": timing[
                    "myokit_cuda_minimum"
                ],
                "cubie_minimum_ms": timing["cubie_minimum"],
                "cubie_over_myokit_cuda": timing[
                    "cubie_over_myokit_cuda"
                ],
                "myokit_cuda_over_cubie": timing[
                    "myokit_cuda_over_cubie"
                ],
                "maximum_absolute_error": accuracy[
                    "maximum_absolute_error"
                ],
                "root_mean_square_error": accuracy[
                    "root_mean_square_error"
                ],
                "allclose": report["allclose"]["result"],
            }
        )
    with (output_dir / "scaling.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# CuBIE vs Myokit-CUDA: Fabbri-Linder scaling\n\n",
        "Both implementations use float32 forward Euler with `dt={0:g}` "
        "for {1} steps ({2:g} seconds total). Compilation and one warmup "
        "solve per trajectory count are excluded; each row is the minimum "
        "of {3} synchronized repeats. CUDA block sizes are {4} for "
        "Myokit-CUDA and {5} for CuBIE.\n\n".format(
            reports[0]["dt"],
            reports[0]["steps"],
            reports[0]["duration"],
            reports[0]["repeats"],
            reports[0]["block_size"]["myokit_cuda"],
            reports[0]["block_size"]["cubie"],
        ),
        "| N | Myokit-CUDA (ms) | CuBIE (ms) | CuBIE / Myokit-CUDA | "
        "CuBIE speedup | Max abs error | RMS error | allclose |\n",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|\n",
    ]
    for row in rows:
        markdown.append(
            "| {trajectories} | {myokit_cuda_minimum_ms:.6f} | "
            "{cubie_minimum_ms:.6f} | {cubie_over_myokit_cuda:.6f} | "
            "{myokit_cuda_over_cubie:.6f}x | "
            "{maximum_absolute_error:.9e} | "
            "{root_mean_square_error:.9e} | {allclose} |\n".format(
                **row
            )
        )
    (output_dir / "comparison.md").write_text(
        "".join(markdown),
        encoding="utf-8",
    )


def trajectory_counts(arguments):
    """Resolve the public single-point or scaling-count selection."""
    if arguments.trajectory_counts is not None:
        counts = arguments.trajectory_counts
    elif arguments.trajectories is not None:
        counts = [arguments.trajectories]
    else:
        counts = [512, 2048, 8192, 32768]
    if any(count <= 0 for count in counts):
        raise ValueError("trajectory counts must be positive")
    if len(set(counts)) != len(counts):
        raise ValueError("trajectory counts must be unique")
    return counts


def failed_trajectory_counts(reports):
    """Return trajectory counts that failed numerical equivalence."""
    return [
        report["trajectories"]
        for report in reports
        if not report["allclose"]["result"]
    ]


def orchestrate(arguments):
    """Run both implementations and write the comparison artifacts."""
    cellml = Path(arguments.cellml).resolve()
    cubie_python = Path(arguments.cubie_python).resolve()
    if not cellml.is_file():
        raise FileNotFoundError(str(cellml))
    if not cubie_python.is_file():
        raise FileNotFoundError(str(cubie_python))

    counts = trajectory_counts(arguments)
    output_dir = (
        Path(arguments.output_dir).resolve()
        / dataset_key()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    with tempfile.TemporaryDirectory(
        prefix="cubie_myokit_fabbri_"
    ) as temporary:
        normalized_cellml = Path(temporary) / "Fabbri_Linder_myokit.cellml"
        cellml_repairs = normalized_fabbri_cellml(
            cellml, normalized_cellml
        )
        model = MyokitCudaModel(
            normalized_cellml,
            block_size=arguments.myokit_block_size,
        )
        for trajectories in counts:
            print(
                "Comparing {0} Fabbri-Linder trajectories..."
                .format(trajectories)
            )
            myokit_ms, myokit_finals, myokit_samples = (
                timed_myokit_solve(
                    model,
                    trajectories,
                    arguments.dt,
                    arguments.steps,
                    arguments.repeats,
                )
            )
            if not np.all(np.isfinite(myokit_finals)):
                raise FloatingPointError(
                    "Myokit-CUDA produced non-finite states at N={0}"
                    .format(trajectories)
                )

            worker_output = (
                Path(temporary)
                / "cubie_result_{0}.npz".format(trajectories)
            )
            run_cubie_subprocess(
                arguments, worker_output, trajectories
            )
            with np.load(worker_output) as data:
                cubie_names = tuple(
                    str(name) for name in data["state_names"]
                )
                cubie_raw = np.array(data["finals"], copy=True)
                cubie_samples = np.array(
                    data["samples_seconds"], copy=True
                )

            cubie_finals, mapped_names = mapped_cubie_states(
                model.state_names,
                cubie_names,
                cubie_raw,
            )
            if not np.all(np.isfinite(cubie_finals)):
                raise FloatingPointError(
                    "CuBIE produced non-finite states at N={0}"
                    .format(trajectories)
                )
            cubie_ms = float(np.min(cubie_samples) * 1000.0)
            detail_dir = (
                output_dir / "N_{0}".format(trajectories)
            )
            report = write_outputs(
                detail_dir,
                arguments,
                mapped_names,
                myokit_finals,
                cubie_finals,
                myokit_ms,
                cubie_ms,
                myokit_samples,
                cubie_samples,
                cellml_repairs,
                trajectories,
            )
            reports.append(report)
            print(
                "Myokit-CUDA: {0:.6f} ms; CuBIE: {1:.6f} ms; "
                "max abs error: {2:.9e}".format(
                    myokit_ms,
                    cubie_ms,
                    report["accuracy"]["maximum_absolute_error"],
                )
            )

    write_scaling_summary(output_dir, reports)
    print("Wrote comparison to {0}".format(output_dir))
    failed_counts = failed_trajectory_counts(reports)
    if failed_counts:
        print(
            "Numerical equivalence failed at trajectory counts: {0}"
            .format(
                ", ".join(str(count) for count in failed_counts)
            ),
            file=sys.stderr,
        )
        return 1
    return 0


def orchestrator_parser():
    """Build the public comparison command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cellml", required=True)
    parser.add_argument("--cubie-python", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "Fabbri_Myokit_CUDA"),
    )
    count_group = parser.add_mutually_exclusive_group()
    count_group.add_argument(
        "--trajectories",
        type=positive_integer,
        default=None,
        help="run one smoke/performance point",
    )
    count_group.add_argument(
        "--trajectory-counts",
        type=positive_integer,
        nargs="+",
        default=None,
        help=(
            "run a scaling comparison (default: 512 2048 8192 32768)"
        ),
    )
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument(
        "--steps", type=positive_integer, default=DEFAULT_STEPS
    )
    parser.add_argument("--repeats", type=positive_integer, default=100)
    parser.add_argument(
        "--myokit-block-size", type=positive_integer, default=128
    )
    parser.add_argument(
        "--cubie-block-size", type=positive_integer, default=64
    )
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    return parser


def worker_parser():
    """Build the internal CuBIE worker parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cellml", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trajectories", type=int, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    return parser


def main(argv=None):
    """Dispatch the public orchestrator or hidden CuBIE worker."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "cubie-worker":
        return cubie_worker(worker_parser().parse_args(argv[1:]))
    return orchestrate(orchestrator_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
