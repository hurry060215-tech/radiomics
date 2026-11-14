"""Build an immutable, reviewable native wheel/CLI snapshot and provenance receipt.

Uses the existing compiler and maturin; never installs, publishes or changes a
version. Run from a clean source commit, with a new output directory outside the
repository or under an ignored artifact directory.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def capture(argv,root,environment):
    return subprocess.run(argv,cwd=root,env=environment,check=True,capture_output=True,text=True).stdout.strip()


def source_state(root,environment):
    return dict(revision=capture(['git','rev-parse','HEAD'],root,environment),
        tree=capture(['git','rev-parse','HEAD^{tree}'],root,environment),
        status=capture(['git','status','--porcelain'],root,environment))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target-dir',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    environment=os.environ.copy()
    before=source_state(root,environment)
    if before['status']:
        raise RuntimeError('An immutable snapshot requires a clean committed source tree.')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    if args.target_dir is not None:
        environment['CARGO_TARGET_DIR']=str(args.target_dir.resolve())
    environment.update(CARGO_PROFILE_RELEASE_OPT_LEVEL='3',CARGO_PROFILE_RELEASE_LTO='thin',CARGO_PROFILE_RELEASE_CODEGEN_UNITS='1')
    rustc=capture(['rustc','-vV'],root,environment)
    host=next(line.split(': ',1)[1] for line in rustc.splitlines() if line.startswith('host: '))
    target=environment.get('CARGO_BUILD_TARGET') or host
    if target!=host:
        raise RuntimeError('This native snapshot tool verifies its built extension on the host; cross-target builds require separate validation.')
    commands=[
        [sys.executable,str(root/'tools/package_native.py'),'--output',str(output/'package-assets')],
        ['cargo','build','--release','-p','ov-radiomics-cli','--locked'],
        [sys.executable,'-m','maturin','build','--release','--locked','--out',str(output/'wheels')],
    ]
    for command in commands:
        subprocess.run(command,cwd=root,env=environment,check=True)
    wheels=list((output/'wheels').glob('*.whl'))
    if len(wheels)!=1:
        raise RuntimeError('Expected exactly one wheel for a native snapshot.')
    wheel=wheels[0]
    target_root=Path(environment.get('CARGO_TARGET_DIR',root/'target'))
    if not target_root.is_absolute():
        target_root=root/target_root
    if environment.get('CARGO_BUILD_TARGET'):
        target_root/=target
    binary='radiomics.exe' if 'windows' in target else 'radiomics'
    cli=output/'cli'/binary;cli.parent.mkdir()
    shutil.copy2(target_root/'release'/binary,cli)
    view=output/'wheel-view';view.mkdir()
    files={}
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            path=(view/member.filename).resolve()
            if not path.is_relative_to(view.resolve()) or Path(member.filename).is_absolute():
                raise ValueError('Wheel member escapes the verification directory.')
            if member.is_dir():
                path.mkdir(parents=True,exist_ok=True);continue
            path.parent.mkdir(parents=True,exist_ok=True)
            data=archive.read(member);path.write_bytes(data)
            files[member.filename]=hashlib.sha256(data).hexdigest()
    # Import the wheel's own extension in a clean subprocess, without changing
    # the development environment or trusting an editable installation.
    code="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import ov_radiomics._native as native;assert Path(native.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve());print(native.IMPLEMENTATION_SHA256)"
    implementation=capture([sys.executable,'-c',code,str(view)],root,environment)
    if len(implementation)!=64 or any(c not in '0123456789abcdef' for c in implementation):
        raise RuntimeError('Built wheel did not report its typed core implementation identity.')
    after=source_state(root,environment)
    if after!=before:
        raise RuntimeError('Source changed during build; artifacts retained for diagnostics, not accepted as a snapshot.')
    receipt=dict(schema='radiomics.native-build.v1',source_revision=before['revision'],source_tree=before['tree'],
        source_tree_clean=True,cargo_lock_sha256=sha256(root/'Cargo.lock'),core_implementation_sha256=implementation,
        rustc_verbose=rustc,rust_target=target,profile=dict(name='release',opt_level=3,lto='thin',codegen_units=1),
        rust_flags={name:environment.get(name) for name in ('RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS')},
        profile_environment={name:environment[name] for name in ('CARGO_PROFILE_RELEASE_OPT_LEVEL','CARGO_PROFILE_RELEASE_LTO','CARGO_PROFILE_RELEASE_CODEGEN_UNITS')},
        build_commands=commands,python_version=sys.version.split()[0],maturin_version=importlib.metadata.version('maturin'),
        wheel=dict(path=str(wheel),sha256=sha256(wheel)),cli=dict(path=str(cli),sha256=sha256(cli)),
        wheel_files_sha256=files,published=False,version_changed=False,
        evidence_scope='Build identity only; correctness, installed-wheel, standards and performance gates require separate reports.')
    (output/'build-manifest.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
    print(output/'build-manifest.json')


if __name__=='__main__':
    main()
