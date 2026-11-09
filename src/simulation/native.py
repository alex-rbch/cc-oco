from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .config import ExperimentConfig


class _NativeStats(ctypes.Structure):
    _fields_ = [
        ("learner_loss", ctypes.c_double),
        ("short_learner_loss", ctypes.c_double),
        ("long_learner_loss", ctypes.c_double),
        ("maximum_weighted_gradient_norm", ctypes.c_double),
        ("number_nonfinite_updates", ctypes.c_int64),
        ("maximum_active", ctypes.c_int64),
        ("saturated_rounds", ctypes.c_int64),
        ("observed_short", ctypes.c_int64),
        ("observed_long", ctypes.c_int64),
        ("weight_count", ctypes.c_int64),
    ]


_LIBRARY: Optional[ctypes.CDLL] = None
_BUILD_ATTEMPTED = False


def _load_library() -> Optional[ctypes.CDLL]:
    global _LIBRARY, _BUILD_ATTEMPTED
    if _BUILD_ATTEMPTED:
        return _LIBRARY
    _BUILD_ATTEMPTED = True
    source = Path(__file__).with_name("native_sim.cpp")
    digest = hashlib.sha256(source.read_bytes() + platform.machine().encode()).hexdigest()[:16]
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    cache = Path(tempfile.gettempdir()) / f"cc-oco-native-{os.getuid()}"
    cache.mkdir(mode=0o700, exist_ok=True)
    output = cache / f"simulation-{digest}{suffix}"
    if not output.exists():
        compiler = shlex.split(os.environ.get("CXX", ""))
        if not compiler:
            executable = next((shutil.which(name) for name in ("clang++", "g++", "c++") if shutil.which(name)), None)
            if executable is None:
                return None
            compiler = [executable]
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache, suffix=suffix, delete=False) as handle:
                temporary = Path(handle.name)
            subprocess.run(
                compiler + [
                    "-O3",
                    "-std=c++17",
                    "-shared",
                    "-fPIC",
                    str(source),
                    "-o",
                    str(temporary),
                ],
                check=True,
                capture_output=True,
            )
            temporary.replace(output)
        except (OSError, subprocess.CalledProcessError):
            return None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    try:
        _LIBRARY = ctypes.CDLL(str(output))
    except OSError:
        return None
    function = _LIBRARY.simulate_delayed
    function.restype = ctypes.c_int
    function.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(_NativeStats),
    ]
    return _LIBRARY


def run_native_core(
    environment: Any,
    config: ExperimentConfig,
    problem_type: str,
    scheduler: str,
    proxy_draws: np.ndarray,
    proxy: Any,
    learning_parameters: Any,
) -> Optional[Dict[str, Any]]:
    library = _load_library()
    if library is None:
        return None
    delays = np.ascontiguousarray(environment.delays, dtype=np.int64)
    is_long = np.ascontiguousarray(environment.gradients.is_long, dtype=np.uint8)
    short_vector = np.ascontiguousarray(environment.gradients.short_vector, dtype=np.float64)
    long_vector = np.ascontiguousarray(environment.gradients.long_vector, dtype=np.float64)
    directions = np.ascontiguousarray(environment.bco_directions, dtype=np.float32)
    draws = np.ascontiguousarray(proxy_draws, dtype=np.int64)
    explicit = environment.gradients.explicit
    explicit_array = (
        np.ascontiguousarray(explicit, dtype=np.float64) if explicit is not None else None
    )
    states = np.empty(len(delays), dtype=np.uint8)
    weights = np.empty(len(delays), dtype=np.float64)
    stats = _NativeStats()
    null_double = ctypes.POINTER(ctypes.c_double)()
    explicit_pointer = (
        explicit_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if explicit_array is not None
        else null_double
    )
    result = library.simulate_delayed(
        len(delays),
        config.dimension,
        config.capacity,
        0 if problem_type == "oco" else 1,
        {"full": 0, "pareto": 1, "uniform": 2}[scheduler],
        delays.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        is_long.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        short_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        long_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        explicit_pointer,
        directions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        draws.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        float(getattr(proxy, "shape", 1.5)),
        float(getattr(proxy, "scale", 1.0)),
        float(getattr(proxy, "p", 1.0)),
        float(environment.gradients.constant_offset),
        float(environment.gradients.quadratic_coefficient),
        config.domain_radius,
        learning_parameters.smoothing_radius
        if problem_type == "bco"
        else config.domain_radius * 0.5,
        learning_parameters.learning_rate,
        states.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(stats),
    )
    if result != 0:
        raise RuntimeError(f"native simulation failed with status {result}")
    return {
        "states": states,
        "weights": weights[: stats.weight_count],
        "learner_loss": stats.learner_loss,
        "short_learner_loss": stats.short_learner_loss,
        "long_learner_loss": stats.long_learner_loss,
        "maximum_weighted_gradient_norm": stats.maximum_weighted_gradient_norm,
        "number_nonfinite_updates": stats.number_nonfinite_updates,
        "maximum_active": stats.maximum_active,
        "saturated_rounds": stats.saturated_rounds,
        "observed_short": stats.observed_short,
        "observed_long": stats.observed_long,
    }
