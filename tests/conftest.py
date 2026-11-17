"""Shared native availability gate; broken or stale binaries remain failures."""

import pytest

from ov_radiomics._backend import resolve_backend, try_native


@pytest.fixture(scope="session")
def native_backend():
    if try_native() is None:
        pytest.skip("Native extension not built.")
    return resolve_backend("rust")
