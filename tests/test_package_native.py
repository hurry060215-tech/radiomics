"""Packaging must verify frozen bytes and refuse unsafe/stale asset replacement."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('package_native',Path(__file__).resolve().parents[1]/'tools/package_native.py')
package=importlib.util.module_from_spec(spec);spec.loader.exec_module(package)


def repository(root):
    (root/'LICENSE').write_bytes(b'Original MIT source license\n')
    for profile in package.PROFILES:
        path=root/'golden'/profile;path.mkdir(parents=True)
        data={'input.json':b'{"unchanged":true}\n'} if profile=='python_golden_v1' else {
            'image.nii.gz':b'original compressed image bytes','mask.nii.gz':b'original compressed mask bytes',
            'LICENSE.md':b'CC-BY-4.0 original license','SOURCE_README.md':b'Alex Zwanenburg original source attribution'}
        for name,value in data.items():(path/name).write_bytes(value)
        manifest=dict(files_sha256={name:hashlib.sha256(value).hexdigest() for name,value in data.items()})
        if profile=='ibsi_digital_v1':manifest.update(license='CC-BY-4.0',author='Alex Zwanenburg')
        (path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    for name,relative in package.PACKAGES.values():
        crate=root/relative;crate.mkdir(parents=True);(crate/'Cargo.toml').write_text('[package]\nname="'+name+'"\n',encoding='utf-8')
    return root


def test_assets_keep_exact_reference_bytes_licenses_and_only_explicit_profiles(tmp_path):
    root=repository(tmp_path)
    # An unrelated/nonredistributable directory is never recursively copied.
    extra=root/'golden'/'private_ct';extra.mkdir();(extra/'patient.nii').write_bytes(b'do not package')
    result=package.prepare(root,['core','cli','python'])
    assert all(row['written']>0 for row in result)
    directory=root/'crates/radiomics-core/package-assets'
    assert (directory/'golden/ibsi_digital_v1/image.nii.gz').read_bytes()==b'original compressed image bytes'
    assert (directory/'LICENSE').read_bytes()==(root/'LICENSE').read_bytes()
    assert b'Alex Zwanenburg' in (directory/'THIRD_PARTY_NOTICES.md').read_bytes()
    assert not (directory/'golden/private_ct').exists()
    assert not (root/'crates/radiomics-python/package-assets/golden').exists()
    assert all(row['written']==0 for row in package.prepare(root,['core','cli','python']))


def test_frozen_hash_mismatch_aborts_before_any_generated_writes(tmp_path):
    root=repository(tmp_path)
    (root/'golden/python_golden_v1/input.json').write_bytes(b'changed fixture')
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        package.prepare(root,['core','cli'])
    assert not (root/'crates/radiomics-core/package-assets').exists()


def test_unknown_stage_files_and_changed_known_files_are_never_silently_overwritten(tmp_path):
    root=repository(tmp_path);package.prepare(root,['core'])
    directory=root/'crates/radiomics-core/package-assets'
    (directory/'README.md').write_bytes(b'changed known generated file')
    with pytest.raises(FileExistsError,match='--overwrite'):
        package.prepare(root,['core'])
    package.prepare(root,['core'],overwrite=True)
    assert b'ov-radiomics-core' in (directory/'README.md').read_bytes()
    (directory/'unrelated.bin').write_bytes(b'preserve this file')
    with pytest.raises(FileExistsError,match='Unknown file'):
        package.prepare(root,['core'],overwrite=True)
    assert (directory/'unrelated.bin').read_bytes()==b'preserve this file'


def test_manifest_traversal_is_rejected_before_reading_outside_sources(tmp_path):
    root=repository(tmp_path)
    path=root/'golden/python_golden_v1/manifest.json'
    manifest=json.loads(path.read_text(encoding='utf-8'));manifest['files_sha256']['../../LICENSE']='0'*64
    path.write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError,match='remain inside'):
        package.prepare(root,['core'])


def test_artifact_only_lock_pruning_rejects_any_dependency_identity_drift():
    original=b'[[package]]\nname = "core"\nversion = "0.1.0"\n\n[[package]]\nname = "serde"\nversion = "1.0.229"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\nchecksum = "abc"\n\n[[package]]\nname = "cli"\nversion = "0.1.0"\n'
    derived=original[:original.rindex(b'[[package]]')]
    assert package.verify_pruned_lock(original,derived)==[('cli','0.1.0',None,None)]
    for changed in [derived.replace(b'1.0.229',b'1.0.230'),derived.replace(b'checksum = "abc"',b'checksum = "def"')]:
        with pytest.raises(ValueError,match='changes a package'):
            package.verify_pruned_lock(original,changed)
