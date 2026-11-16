"""Explicit execution selection and a separate read-only native availability probe."""

from importlib import import_module
from types import ModuleType
from typing import Literal, Optional, overload


def try_native() -> Optional[ModuleType]:
    """Read availability without executing any optional reference computation."""
    try:
        return import_module("ov_radiomics._native")
    except ModuleNotFoundError as exc:
        # Only an absent extension permits fallback. A broken DLL/dependency must surface.
        if exc.name != "ov_radiomics._native":
            raise
        return None


@overload
def resolve_backend(backend: Literal["rust", "auto"]) -> ModuleType: ...


@overload
def resolve_backend(backend: Literal["python"]) -> None: ...


@overload
def resolve_backend(backend: str) -> Optional[ModuleType]: ...


def resolve_backend(backend: str) -> Optional[ModuleType]:
    if backend not in ("auto", "python", "rust"):
        raise ValueError("backend must be 'auto', 'python', or 'rust'.")
    if backend == "python":
        return None
    native = try_native()
    if native is None:
        raise ImportError(
            "Rust extension is not built. Install a compiled wheel or run maturin develop "
            "--release; choose backend=python explicitly for an optional reference backend. "
            "auto never falls back."
        )
    return native


def backend_info() -> dict[str, Optional[str]]:
    """Report the actual default metric backend; does not change process-wide settings."""
    native = try_native()
    return {
        "metrics_backend": "rust" if native is not None else "unavailable",
        "rust_core_version": native.CORE_VERSION if native is not None else None,
    }
