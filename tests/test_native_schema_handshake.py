"""Old binaries fail with actionable guidance without hiding validation errors."""

from types import ModuleType

import pytest

from ov_radiomics import _backend, _native_common


def test_old_binary_rejected_before_dispatch(monkeypatch):
    monkeypatch.setattr(_backend, "try_native", lambda: ModuleType("old_native"))
    with pytest.raises(ImportError, match="Rebuild with: python -m maturin develop --release"):
        _backend.resolve_backend("rust")


@pytest.mark.parametrize(
    "message, guidance",
    [
        ("Schema: unknown field `threads`", True),
        ("Mask has no selected voxels", False),
    ],
)
def test_request_schema_diagnostic_preserves_real_errors(monkeypatch, message, guidance):
    native = ModuleType("mock_native")

    def fail(payload):
        raise ValueError(message)

    native.request_json = fail
    monkeypatch.setattr(_native_common, "resolve_backend", lambda backend: native)
    with pytest.raises(ValueError) as error:
        _native_common.request("array_features", threads=1)
    assert ("Rebuild with" in str(error.value)) == guidance
    assert message in str(error.value)


def test_matching_binary_accepted(monkeypatch):
    native = ModuleType("current_native")
    native.SCHEMA_VERSION = _backend.NATIVE_SCHEMA_VERSION
    native.API_REVISION = _backend.NATIVE_API_REVISION
    monkeypatch.setattr(_backend, "try_native", lambda: native)
    assert _backend.resolve_backend("rust") is native
