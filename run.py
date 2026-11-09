import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the capacity ablation for convex and strongly convex losses."
    )
    parser.add_argument("--horizon", type=int, help="rounds per run, at least 10000 (default: 1000000)")
    parser.add_argument("--seeds", type=int, help="paired repetitions, starting at seed 100 (default: 100)")
    parser.add_argument("--workers", type=int, default=1, help="parallel repetitions (default: 1)")
    parser.add_argument("--output", type=Path, default=Path("results"), help="results directory (default: results)")
    parser.add_argument("--plot-only", action="store_true", help="plot completed results using their saved settings")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.horizon is not None and args.horizon < 10000:
        parser.error("--horizon must be at least 10000")
    if args.seeds is not None and args.seeds < 1:
        parser.error("--seeds must be positive")
    if args.plot_only and (args.horizon is not None or args.seeds is not None):
        parser.error("--plot-only uses saved settings; omit --horizon and --seeds")
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(variable, "1")
    from src.experiment import run

    try:
        run(args.output, args.horizon, args.seeds, args.workers, args.plot_only)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nInterrupted. Completed repetitions are saved; rerun the command to continue.\n")


if __name__ == "__main__":
    main()
