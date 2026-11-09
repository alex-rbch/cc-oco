import csv
import itertools
import json
import math
import multiprocessing
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from .simulation.actual_delays import delay_scales_for_config
from .simulation.config import ExperimentConfig
from .simulation.environment import build_environment, run_simulation
from .simulation.native import _load_library

CAPACITIES = (20, 40, 80, 160)
FAMILIES = {"convex": 0.0, "strongly_convex": 0.25}
PROBLEMS = ("oco", "bco")
SCHEDULERS = ("uniform", "pareto", "full")
FIELDS = (
    "loss_family", "seed", "problem_type", "scheduler", "capacity", "horizon",
    "total_regret", "normalized_regret", "learner_loss", "comparator_loss",
    "realized_mean_delay", "sigma_max", "loss_delay_threshold",
    "maximum_active_set_size", "fixed_learning_rate", "fixed_smoothing_radius",
    "fraction_observed", "fraction_proxy_rejected", "fraction_proxy_expired",
    "fraction_capacity_rejected", "fraction_horizon_unresolved",
    "short_round_observation_rate", "long_round_observation_rate",
    "mean_importance_weight", "mean_squared_importance_weight",
    "maximum_importance_weight", "number_nonfinite_updates", "number_clipped_updates",
)


def paper_config(horizon=1000000, seeds=100):
    config = ExperimentConfig(
        horizon=horizon, capacity=CAPACITIES[0], target_mean_delays=(400,),
        changing_direction_bias=0.95, loss_coupling="delayed_indicator_changing",
        loss_delay_quantiles=((400, 0.4),), final_seeds=tuple(range(100, 100 + seeds)),
    )
    config.validate()
    return config


def atomic_text(path, contents):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(contents)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_rows(path, rows):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(sorted(rows, key=row_key))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def row_key(row):
    return (int(row["seed"]), row["loss_family"], row["problem_type"], int(row["capacity"]), row["scheduler"])


def expected_keys(seed):
    return {(seed, family, problem, capacity, scheduler)
            for family, problem, capacity, scheduler in itertools.product(FAMILIES, PROBLEMS, CAPACITIES, SCHEDULERS)}


def read_rows(path, config):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected CSV columns in {path}")
        rows = list(reader)
    allowed = set().union(*(expected_keys(seed) for seed in config.final_seeds))
    seen = set()
    for row in rows:
        key = row_key(row)
        if key not in allowed or key in seen or int(row["horizon"]) != config.horizon:
            raise ValueError(f"Invalid or duplicate run in {path}: {key}")
        seen.add(key)
        regret, normalized, learner, comparator = (
            float(row[field]) for field in ("total_regret", "normalized_regret", "learner_loss", "comparator_loss")
        )
        if not all(math.isfinite(value) for value in (regret, normalized, learner, comparator)):
            raise ValueError(f"Nonfinite loss or regret in {path}: {key}")
        if not math.isclose(regret, learner - comparator, abs_tol=1e-7) or not math.isclose(regret, normalized * config.horizon, abs_tol=1e-7):
            raise ValueError(f"Inconsistent regret in {path}: {key}")
        if row["scheduler"] != "full" and int(row["maximum_active_set_size"]) > int(row["capacity"]):
            raise ValueError(f"Capacity exceeded in {path}: {key}")
    return rows


def complete_seeds(rows, seeds):
    groups = {seed: set() for seed in seeds}
    for row in rows:
        groups[int(row["seed"])].add(row_key(row))
    return {seed for seed, keys in groups.items() if keys == expected_keys(seed)}


def prepare_settings(output, horizon, seeds, plot_only=False):
    path = output / "config.json"
    if path.exists():
        settings = json.loads(path.read_text())
        if settings.get("schema_version") != 2 or settings.get("capacities") != list(CAPACITIES) or settings.get("loss_families") != FAMILIES:
            raise ValueError("Saved experiment settings are incompatible with this runner")
        config = ExperimentConfig.from_dict(settings["base_configuration"])
        if config.pareto_proxy_shape != 1.0:
            raise ValueError("Saved results use a different Lomax exponent. Choose a separate --output directory.")
        if (horizon is not None and horizon != config.horizon) or (seeds is not None and seeds != len(config.final_seeds)):
            raise ValueError("This output directory has different settings. Choose a separate --output directory.")
        return config, settings
    if plot_only:
        raise ValueError(f"No saved settings in {output}; run an experiment first")
    if (output / "runs.csv").exists():
        raise ValueError(f"Found runs.csv without config.json in {output}")
    config = paper_config(horizon if horizon is not None else 1000000, seeds if seeds is not None else 100)
    print("Preparing delay distribution...", flush=True)
    scale = delay_scales_for_config(config)[400.0]
    settings = dict(schema_version=2, base_configuration=config.to_dict(),
                    capacities=list(CAPACITIES), loss_families=FAMILIES, delay_scale=scale)
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(path, json.dumps(settings, indent=2, allow_nan=False) + "\n")
    return config, settings


def result_row(diagnostics, family, seed, capacity, horizon):
    values = dict(diagnostics, loss_family=family, seed=seed, capacity=capacity, horizon=horizon)
    return {field: "N/A" if isinstance(values[field], float) and not math.isfinite(values[field]) else values[field]
            for field in FIELDS}


def run_seed(config, scale, seed):
    rows = []
    for family, mixture in FAMILIES.items():
        base = replace(config, quadratic_mixture=mixture)
        environment = build_environment(base, 400, seed, scale)
        full = {problem: dict(run_simulation(environment, base, problem, "full").diagnostics)
                for problem in PROBLEMS}
        for capacity in CAPACITIES:
            current = replace(base, capacity=capacity)
            current.validate()
            diagnostics = dict(environment.diagnostics, capacity=capacity,
                               sigma_max_over_capacity=environment.sigma_max / capacity)
            paired = replace(environment, diagnostics=diagnostics)
            for problem in PROBLEMS:
                rows.append(result_row(full[problem], family, seed, capacity, config.horizon))
                for scheduler in ("uniform", "pareto"):
                    outcome = run_simulation(paired, current, problem, scheduler)
                    rows.append(result_row(outcome.diagnostics, family, seed, capacity, config.horizon))
    return rows


def run(output, horizon=None, seeds=None, workers=1, plot_only=False):
    from .plotting import plot_results

    output = Path(output)
    config, settings = prepare_settings(output, horizon, seeds, plot_only)
    rows = read_rows(output / "runs.csv", config)
    completed = complete_seeds(rows, config.final_seeds)
    missing = [seed for seed in config.final_seeds if seed not in completed]
    if plot_only and missing:
        raise ValueError(f"{len(missing)} repetitions are incomplete. Rerun without --plot-only to finish them.")
    if missing:
        if _load_library() is None:
            raise RuntimeError("A C++17 compiler is required. Install clang++ or g++, or set CXX to your compiler.")
        rows = [row for row in rows if int(row["seed"]) in completed]
        print(f"Reusing {len(completed)}/{len(config.final_seeds)} completed repetitions.", flush=True)
        started = time.monotonic()

        def record(batch):
            rows.extend(batch)
            completed.add(int(batch[0]["seed"]))
            write_rows(output / "runs.csv", rows)
            print(f"Completed {len(completed)}/{len(config.final_seeds)} repetitions ({time.monotonic() - started:.1f}s).", flush=True)

        if workers == 1:
            for seed in missing:
                record(run_seed(config, settings["delay_scale"], seed))
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(missing)), mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = [pool.submit(run_seed, config, settings["delay_scale"], seed) for seed in missing]
                try:
                    for future in as_completed(futures):
                        record(future.result())
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
    else:
        print(f"Reusing all {len(config.final_seeds)} completed repetitions.", flush=True)
    plot_results(rows, config, output)
    print(f"Saved {output / 'figure3.pdf'} and {output / 'figure3.png'}", flush=True)
