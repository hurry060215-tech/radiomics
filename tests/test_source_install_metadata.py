"""PEP 517 metadata must not depend on ignored, generated crate assets."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_binding_readme_exists_in_the_source_and_is_included():
    root = Path(__file__).resolve().parents[1]
    crate = root / "crates" / "radiomics-python"
    with (crate / "Cargo.toml").open("rb") as handle:
        package = tomllib.load(handle)["package"]
    relative = Path(package["readme"])
    assert not relative.is_absolute()
    assert ".." not in relative.parts
    assert "package-assets" not in relative.parts
    assert (crate / relative).is_file()
    assert "/" + relative.as_posix() in package["include"]
