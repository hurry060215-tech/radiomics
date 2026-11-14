"""Prepare hash-verified crate-local assets and optional development .crate files.

The canonical Golden directories remain the only maintained fixture copies.
Generated package-assets are ignored by git and explicitly included by Cargo.
This tool never publishes, changes versions, edits manifests or updates Golden.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import os
import shutil
import tarfile
import re
import copy
import gzip
import io
from datetime import datetime, timezone

PACKAGES={
    'core':('ov-radiomics-core','crates/radiomics-core'),
    'cli':('ov-radiomics-cli','crates/radiomics-cli'),
    'python':('ov-radiomics-python','crates/radiomics-python'),
}
PROFILES=('python_golden_v1','ibsi_digital_v1')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def contained(root,relative):
    """Reject absolute/traversing paths and existing symlinks escaping the root."""
    relative=Path(relative)
    root=Path(root).resolve()
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Asset paths must be relative and remain inside their declared root.')
    result=(root/relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Resolved asset path escapes its declared root.')
    return result


def frozen_assets(root):
    """Read and validate every allowlisted byte before writing any staged output."""
    root=Path(root).resolve();assets={};manifests={}
    for profile in PROFILES:
        directory=contained(root,'golden/'+profile)
        raw=(directory/'manifest.json').read_bytes();manifest=json.loads(raw)
        checks=manifest.get('files_sha256')
        if not isinstance(checks,dict) or not checks:
            raise ValueError('Golden manifest must declare a nonempty files_sha256 mapping: '+profile)
        if profile=='ibsi_digital_v1' and (manifest.get('license')!='CC-BY-4.0' or manifest.get('author')!='Alex Zwanenburg'):
            raise ValueError('IBSI digital source must retain frozen CC-BY-4.0 and Alex Zwanenburg attribution.')
        if profile=='ibsi_digital_v1' and not {'LICENSE.md','SOURCE_README.md','image.nii.gz','mask.nii.gz'} <= set(checks):
            raise ValueError('IBSI stage requires its license, attribution source and unchanged image/mask bytes.')
        manifests[profile]=dict(sha256=digest(raw),license=manifest.get('license',manifest.get('input_license')),
            author=manifest.get('author'),declared_files=len(checks))
        assets['golden/'+profile+'/manifest.json']=raw
        for name,expected in checks.items():
            if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
                raise ValueError('Frozen SHA256 must be a lowercase 64-digit hexadecimal string.')
            data=contained(directory,name).read_bytes()
            if digest(data)!=expected:
                raise ValueError('Frozen Golden SHA256 mismatch: '+profile+'/'+name)
            assets['golden/'+profile+'/'+Path(name).as_posix()]=data
    return assets,manifests


def _readme(kind):
    description={
        'core':'Shared native image, feature, table, model, evaluation and lightweight pathology algorithms.',
        'cli':'Independent Rust command line workflows; no Python runtime is required.',
        'python':'Thin PyO3 bindings to the shared native core; this crate is not published independently.',
    }[kind]
    datasets='This archive includes frozen synthetic and IBSI digital test assets under `golden/`. Their original manifests, license and attribution files are retained.\n' if kind!='python' else 'This binding archive does not duplicate the core/CLI Golden test datasets.\n'
    return ('# '+PACKAGES[kind][0]+'\n\n'+description+'\n\n'
        'This is a development source artifact. Packaging is not a formal publication, clinical validation or proof that every release gate passed.\n\n'
        'Project source and current support/limitations: [hurry060215-tech/radiomics](https://github.com/hurry060215-tech/radiomics).\n\n'
        'The software source is MIT licensed; see [LICENSE](LICENSE). Test data and references retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No blanket MIT license is asserted over third-party test data.\n\n'+datasets).encode('utf-8')


def _notices(kind):
    text=('# Licensing and attribution\n\nSoftware source: MIT; the original repository LICENSE is copied unchanged. Dependency code retains the licenses declared by its own packages.\n\n')
    if kind!='python':
        text+=('Original synthetic Golden inputs/reference outputs are dedicated to CC0-1.0 as recorded in `golden/python_golden_v1/manifest.json`; reference/specification source retains its own license and provenance.\n\n'
            'IBSI-1 digital phantom: Alex Zwanenburg, CC-BY-4.0. The original image/mask bytes are unmodified. See `golden/ibsi_digital_v1/LICENSE.md`, `SOURCE_README.md` and `manifest.json` for terms, attribution, source revisions and input SHA256 values.\n\n'
            'The package contains only the two explicitly allowlisted profiles. It does not contain noncommercial CT references, patient images or data with unclear redistribution permission.\n')
    else:
        text+='No third-party image/Golden datasets are copied into this binding archive.\n'
    return text.encode('utf-8')


def prepare(root,kinds,*,overwrite=False):
    root=Path(root).resolve()
    fixtures,manifests=frozen_assets(root)
    license_bytes=(root/'LICENSE').read_bytes()
    plans=[]
    for kind in kinds:
        if kind not in PACKAGES:
            raise ValueError('Unknown native package kind: '+str(kind))
        name,relative=PACKAGES[kind]
        crate=contained(root,relative)
        if not (crate/'Cargo.toml').is_file():
            raise ValueError('Expected native crate manifest is missing: '+relative)
        directory=contained(crate,'package-assets')
        if (crate/'package-assets').is_symlink() or directory.exists() and not directory.is_dir():
            raise FileExistsError('Generated asset root must be a regular directory, not a file or symlink.')
        files=dict(fixtures) if kind!='python' else {}
        files.update({'README.md':_readme(kind),'LICENSE':license_bytes,'THIRD_PARTY_NOTICES.md':_notices(kind)})
        inventory=dict(schema='radiomics.package_assets.v1',package=name,
            canonical_source='repository Golden manifests; no maintained duplicate dataset',
            profiles=manifests if kind!='python' else {},
            files_sha256={path:digest(data) for path,data in sorted(files.items())})
        files['stage_manifest.json']=(json.dumps(inventory,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
        if directory.exists():
            # rglob sees ordinary files; explicitly reject any directory symlink too.
            for existing in directory.rglob('*'):
                if existing.is_symlink():
                    raise ValueError('Generated asset trees may not contain symlinks.')
                if existing.is_file() and existing.relative_to(directory).as_posix() not in files:
                    raise FileExistsError('Unknown file in generated asset tree; no recursive deletion: '+str(existing))
        writes=[]
        for path,data in files.items():
            destination=contained(directory,path)
            if destination.exists():
                if not destination.is_file():
                    raise FileExistsError('Generated asset destination is not a regular file: '+str(destination))
                if destination.read_bytes()==data:
                    continue
                if not overwrite:
                    raise FileExistsError('Generated asset differs; pass --overwrite explicitly: '+str(destination))
            writes.append((destination,data))
        plans.append((name,directory,files,writes))
    # All manifests, destinations and collisions were checked before any writes.
    result=[]
    for name,directory,files,writes in plans:
        for destination,data in writes:
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('wb' if overwrite else 'xb') as output:
                output.write(data)
        # Validate reused/generated bytes too; a package never relies on row count.
        for path,data in files.items():
            if contained(directory,path).read_bytes()!=data:
                raise ValueError('Generated asset verification failed: '+path)
        result.append(dict(package=name,directory=str(directory),files=len(files),written=len(writes),
            stage_manifest_sha256=digest(files['stage_manifest.json'])))
    return result


def lock_nodes(data):
    """Read identity fields from Cargo's generated lock format, without rewriting it."""
    text=data.decode('utf-8');nodes=set()
    for block in re.split(r'(?m)^\[\[package\]\]\s*$',text)[1:]:
        fields={}
        for line in block.splitlines():
            if line.startswith('['):break
            match=re.match(r'^(name|version|source|checksum)\s*=\s*(".*")\s*$',line)
            if match:
                fields[match[1]]=json.loads(match[2])
        if not {'name','version'}<=set(fields):
            raise ValueError('Unsupported generated Cargo.lock package identity.')
        node=tuple(fields.get(field) for field in ('name','version','source','checksum'))
        if node in nodes:raise ValueError('Duplicate Cargo.lock package identity.')
        nodes.add(node)
    if not nodes:raise ValueError('No generated Cargo.lock packages found.')
    return nodes


def verify_pruned_lock(original,derived):
    before,after=lock_nodes(original),lock_nodes(derived)
    if not after<=before:
        raise ValueError('Derived lock graph adds or changes a package/version/source/checksum; refusing the artifact.')
    return sorted(before-after,key=lambda row:tuple('' if value is None else value for value in row))


def logged(argv,cwd,output,label,report):
    result=subprocess.run(argv,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    path=output/(label+'.log');path.write_bytes(result.stdout)
    report['steps'].append(dict(operation=label,argv=argv,exit_code=result.returncode,
        log=path.name,log_sha256=digest(result.stdout)))
    if result.returncode:
        report['status']='failed'
        raise RuntimeError('Command failed; see '+str(path))
    return result.stdout


def derive_sdist(root,output,cargo,target,report):
    """Prune only artifact-local unused workspace nodes and verify locked builds."""
    root_lock=(root/'Cargo.lock').read_bytes()
    original_directory=output/'original-sdist';original_directory.mkdir()
    logged([sys.executable,'-m','maturin','sdist','--out',str(original_directory)],root,output,'sdist-create',report)
    sources=list(original_directory.glob('*.tar.gz'))
    if len(sources)!=1:raise ValueError('Exactly one original maturin source archive is required.')
    original=sources[0];original_bytes=original.read_bytes()
    unpacked=output/'sdist-source';unpacked.mkdir()
    with tarfile.open(original) as source:
        members=source.getmembers()
        tops={Path(member.name).parts[0] for member in members}
        if len(tops)!=1:raise ValueError('Unexpected multiple-root sdist layout.')
        for member in members:
            if not (member.isfile() or member.isdir()) or not (unpacked/member.name).resolve().is_relative_to(unpacked.resolve()):
                raise ValueError('Unsupported or escaping source archive member.')
        source.extractall(unpacked)
    source_root=unpacked/next(iter(tops))
    lock=source_root/'Cargo.lock';before=lock.read_bytes()
    logged(cargo+['metadata','--offline','--format-version','1'],source_root,output,'sdist-lock-prune',report)
    after=lock.read_bytes();removed=verify_pruned_lock(before,after)
    if (root/'Cargo.lock').read_bytes()!=root_lock:raise ValueError('Repository lockfile changed unexpectedly.')
    derived=output/(original.name[:-7]+'-derived.tar.gz')
    with original.open('rb') as raw_source,tarfile.open(fileobj=raw_source,mode='r:gz') as source:
        with derived.open('xb') as raw_output,gzip.GzipFile(filename='',mode='wb',fileobj=raw_output,mtime=0) as compressed:
            with tarfile.open(fileobj=compressed,mode='w',format=tarfile.PAX_FORMAT) as destination:
                for member in source.getmembers():
                    member=copy.copy(member)
                    data=after if member.name==source_root.name+'/Cargo.lock' else source.extractfile(member).read() if member.isfile() else None
                    if data is not None:
                        member.size=len(data)
                        member.pax_headers=member.pax_headers.copy()
                        if 'size' in member.pax_headers:member.pax_headers['size']=str(len(data))
                    destination.addfile(member,None if data is None else io.BytesIO(data))
    # Byte-check every archive member: only the artifact-local lock is different.
    with tarfile.open(original) as a,tarfile.open(derived) as b:
        original_members={m.name:m for m in a.getmembers()};derived_members={m.name:m for m in b.getmembers()}
        if set(original_members)!=set(derived_members):raise ValueError('Derived source archive members changed.')
        for name,member in original_members.items():
            if member.isfile():
                expected=after if name==source_root.name+'/Cargo.lock' else a.extractfile(name).read()
                if b.extractfile(name).read()!=expected:raise ValueError('Unexpected source member byte change: '+name)
    result=dict(original_path=str(original.relative_to(output)),original_sha256=digest(original_bytes),
        original_lock_sha256=digest(before),derived_lock_sha256=digest(after),
        derived_path=derived.name,derived_sha256=digest(derived.read_bytes()),
        reason='Maturin excludes the unrelated CLI workspace member; only unused artifact-local lock nodes are removed.',
        removed_nodes=[dict(zip(('name','version','source','checksum'),node)) for node in removed],
        retained_nodes_unchanged=True,repository_lock_unchanged=True,locked_check='pending',wheel_build='pending')
    report['sdist']=result
    logged(cargo+['check','--workspace','--all-targets','--locked','--target-dir',str(target)],source_root,output,'sdist-locked-check',report)
    result['locked_check']='passed'
    # rustup run selects the requested toolchain only for this subprocess.
    command=(['rustup','run',cargo[1][1:]] if len(cargo)>1 and cargo[1].startswith('+') else [])
    wheels=output/'sdist-wheels'
    logged(command+[sys.executable,'-m','maturin','build','--release','--locked','--target-dir',str(target),'--out',str(wheels)],source_root,output,'sdist-locked-wheel',report)
    result['wheel_build']='passed'
    result['wheels']=[dict(path=str(path.relative_to(output)),sha256=digest(path.read_bytes())) for path in wheels.glob('*.whl')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--crate',action='append',choices=tuple(PACKAGES),dest='kinds')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--package',action='store_true',help='Also run Cargo package with its normal build verification; never publish.')
    parser.add_argument('--allow-dirty',action='store_true')
    parser.add_argument('--toolchain',help='Explicit already installed Rust toolchain, for example 1.85.0.')
    parser.add_argument('--target-dir',type=Path,help='Task-owned Cargo build cache; otherwise use CARGO_TARGET_DIR or output/target.')
    parser.add_argument('--test',action='store_true',help='After packaging, actually run all targets from Cargo\'s unpacked archive.')
    parser.add_argument('--local-core',type=Path,help='Unpacked core for standalone development verification; package preparation patches the existing workspace core to preserve its lock.')
    parser.add_argument('--sdist',action='store_true',help='Also build/check an artifact-only pruned-lock Python sdist and locked wheel.')
    args=parser.parse_args()
    if args.test and not args.package:
        parser.error('--test requires --package')
    if args.local_core and not args.package:
        parser.error('--local-core applies only to --package development verification')
    root=args.root.resolve();kinds=list(dict.fromkeys(args.kinds or list(PACKAGES)))
    output=args.output or root/'artifacts'/('native-packages-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output=output.resolve()
    if args.sdist and output.exists():
        raise FileExistsError('Source derivation preserves its original archive and requires a fresh output directory, including with --overwrite.')
    if output.exists() and not args.overwrite:
        raise FileExistsError('Output exists; choose a new location or pass --overwrite explicitly.')
    stage_kinds=list(dict.fromkeys(kinds+(['core','python'] if args.sdist else [])))
    staged=prepare(root,stage_kinds,overwrite=args.overwrite)
    output.mkdir(parents=True,exist_ok=True)
    report=dict(schema='radiomics.native_package_preparation.v1',staged=staged,steps=[],status='prepared',published=False)
    cargo=['cargo']+(['+'+args.toolchain] if args.toolchain else [])
    target=(args.target_dir or Path(os.environ.get('CARGO_TARGET_DIR',str(output/'target')))).resolve()
    package_cargo=list(cargo)
    standalone_cargo=list(cargo)
    if args.local_core:
        local=args.local_core.resolve()
        if not (local/'Cargo.toml').is_file():raise ValueError('Local core override must be an actual crate directory.')
        workspace_core=contained(root,PACKAGES['core'][1])
        package_cargo+=['--config','patch.crates-io.ov-radiomics-core.path='+json.dumps(workspace_core.as_posix())]
        standalone_cargo+=['--config','patch.crates-io.ov-radiomics-core.path='+json.dumps(local.as_posix())]
        report['development_dependency_override']=dict(package_preparation_core_path=str(workspace_core),standalone_core_path=str(local),manifest_sha256=digest((local/'Cargo.toml').read_bytes()),
            scope='explicit local registry patch for development verification; no registry publication or independent registry installation is asserted')
    if args.package:
        metadata=subprocess.run(package_cargo+['metadata','--no-deps','--format-version','1','--locked'],cwd=root,check=True,capture_output=True)
        versions={row['name']:row['version'] for row in json.loads(metadata.stdout)['packages']}
        report['status']='packaged'
        for kind in kinds:
            name=PACKAGES[kind][0]
            for action,extra in [('list',['--list']),('build',[])]:
                argv=package_cargo+['package','-p',name,'--locked','--target-dir',str(target)]+extra+(['--allow-dirty'] if args.allow_dirty else [])
                completed=subprocess.run(argv,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                log=output/(name+'-'+action+'.log');log.write_bytes(completed.stdout)
                report['steps'].append(dict(package=name,operation=action,argv=argv,exit_code=completed.returncode,
                    log=log.name,log_sha256=digest(completed.stdout)))
                if completed.returncode:
                    report['status']='failed';break
            if completed.returncode:
                continue
            stem=name+'-'+versions[name]
            archive=target/'package'/(stem+'.crate')
            expected=json.loads((root/PACKAGES[kind][1]/'package-assets/stage_manifest.json').read_text(encoding='utf-8'))
            with tarfile.open(archive) as source:
                for relative,sha256 in expected['files_sha256'].items():
                    member=source.extractfile(stem+'/package-assets/'+relative)
                    if member is None or digest(member.read())!=sha256:
                        raise ValueError('Packaged asset bytes do not match the verified stage: '+relative)
            artifact=output/archive.name
            with archive.open('rb') as source,artifact.open('wb' if args.overwrite else 'xb') as destination:
                shutil.copyfileobj(source,destination)
            report.setdefault('artifacts',[]).append(dict(package=name,path=artifact.name,sha256=digest(artifact.read_bytes()),
                assets_verified=True,standalone_tests='pending'))
            if args.test:
                argv=standalone_cargo+['test','--manifest-path',str(target/'package'/stem/'Cargo.toml'),'--all-targets','--locked','--target-dir',str(target)]
                completed=subprocess.run(argv,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                log=output/(name+'-standalone-tests.log');log.write_bytes(completed.stdout)
                report['steps'].append(dict(package=name,operation='standalone_tests',argv=argv,exit_code=completed.returncode,
                    log=log.name,log_sha256=digest(completed.stdout)))
                report['artifacts'][-1]['standalone_tests']='passed' if completed.returncode==0 else 'failed'
                if completed.returncode:report['status']='failed'
    if args.sdist:
        try:
            derive_sdist(root,output,cargo,target,report)
            if report['status']!='failed':report['status']='built'
        except Exception as exc:
            report.update(status='failed',sdist_error=f'{type(exc).__name__}: {exc}')
    (output/'preparation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 1 if report['status']=='failed' else 0


if __name__=='__main__':
    sys.exit(main())
