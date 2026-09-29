"""ctypes interface to upstream MJPC, with an explicit same-MuJoCo ABI check."""

from __future__ import annotations

import ctypes
from pathlib import Path


class MJPCController:
    def __init__(self, library: Path, model_path: Path, config: dict):
        import mujoco
        import numpy as np

        self._np = np
        self._lib = ctypes.CDLL(str(library.resolve()))
        lib = self._lib
        ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
        lib.go1_mjpc_error.restype = ctypes.c_char_p
        lib.go1_mjpc_revision.restype = ctypes.c_char_p
        lib.go1_mjpc_mujoco_version.restype = ctypes.c_int
        lib.go1_mjpc_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.go1_mjpc_create.restype = ctypes.c_void_p
        lib.go1_mjpc_destroy.argtypes = [ctypes.c_void_p]
        lib.go1_mjpc_destroy.restype = None
        lib.go1_mjpc_action.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ptr,
            ptr,
            ptr,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.go1_mjpc_action.restype = ctypes.c_int
        lib.go1_mjpc_residual.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ptr,
            ptr,
            ptr,
            ptr,
        ]
        lib.go1_mjpc_residual.restype = ctypes.c_int
        if lib.go1_mjpc_mujoco_version() != mujoco.mj_version():
            raise RuntimeError(
                "MuJoCo ABI mismatch. Rebuild MJPC with this Python environment."
            )
        self.revision = lib.go1_mjpc_revision().decode()
        from go1_benchmark.mjpc_model import MJPC_REVISION

        if self.revision != MJPC_REVISION:
            raise RuntimeError("MJPC revision mismatch; rebuild the native library")
        p = config["planner"]
        self._handle = lib.go1_mjpc_create(
            str(model_path.resolve()).encode(),
            p["threads"],
            p["iterations_per_control"],
        )
        if not self._handle:
            raise RuntimeError(lib.go1_mjpc_error().decode())

    def _array(self, value, size):
        array = self._np.ascontiguousarray(value, dtype=self._np.float64)
        if array.shape != (size,) or not self._np.isfinite(array).all():
            raise ValueError(f"Expected a finite vector of length {size}")
        return array

    def action(self, data):
        if not self._handle:
            raise RuntimeError("MJPC controller is closed")
        action = self._np.empty(12, dtype=self._np.float64)
        cost = ctypes.c_double()
        status = self._lib.go1_mjpc_action(
            self._handle,
            data.time,
            self._array(data.qpos, 19),
            self._array(data.qvel, 18),
            action,
            ctypes.byref(cost),
        )
        if status or not self._np.isfinite(action).all():
            raise RuntimeError(self._lib.go1_mjpc_error().decode())
        return action, cost.value

    def residual(self, data):
        from go1_benchmark.mjpc_model import TERMS

        residual = self._np.empty(sum(TERMS.values()), dtype=self._np.float64)
        if not self._handle:
            raise RuntimeError("MJPC controller is closed")
        status = self._lib.go1_mjpc_residual(
            self._handle,
            data.time,
            self._array(data.qpos, 19),
            self._array(data.qvel, 18),
            self._array(data.ctrl, 12),
            residual,
        )
        if status:
            raise RuntimeError(self._lib.go1_mjpc_error().decode())
        return residual

    def close(self):
        if self._handle:
            self._lib.go1_mjpc_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
