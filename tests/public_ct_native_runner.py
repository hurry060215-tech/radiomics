"""Post-review independent PAT1 API/CLI/matrix acceptance, never a reference writer.

Run in the native-only installed-wheel environment. CT inputs and all native raw
results stay in an external fresh artifact directory. No timings or speed claims.
"""
import argparse
import csv
import hashlib
import importlib.abc
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PROFILE=ROOT/'golden/public_ct_v1'
FORBIDDEN={'radiomics','SimpleITK','scipy','sklearn','statsmodels','joblib','omicverse','ov','wsidata','seaborn'}


class RejectReference(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise AssertionError('Native acceptance imported an optional/reference backend: '+fullname)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path,value):
    path.write_bytes((json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))


def numeric(actual,expected,name,case,backend):
    actual=float(actual)
    if isinstance(expected,str):
        passed=math.isnan(actual) if expected=='NaN' else actual==float(expected)
        return dict(backend=backend,case=case,feature=name,reference=expected,actual=expected if passed else str(actual),passed=passed,state='nonfinite')
    if name.endswith(('_index','_voxels','VoxelCount')):
        allowed=0.
    elif 'Centroid' in name:
        allowed=1e-6
    else:
        allowed=1e-10+1e-8*abs(expected)
    error=abs(actual-expected)
    return dict(backend=backend,case=case,feature=name,reference=expected,actual=actual if math.isfinite(actual) else str(actual),
        absolute_error=error if math.isfinite(error) else str(error),allowed_error=allowed,
        passed=math.isfinite(actual) and error<=allowed,state='finite')


def check_geometry(actual,expected):
    import numpy as np
    if actual['unit']!='millimeter' or actual['ndim']!=expected['ndim'] or actual['shape']!=expected['shape_xyz']:
        raise AssertionError('Native geometry dimension/unit differs from frozen CT reference.')
    a=np.asarray(actual['affine'],dtype=float);b=np.asarray(expected['affine_ras_mm'],dtype=float)
    shape=expected['shape_xyz'];errors=[]
    for z in (-.5,shape[2]-.5):
        for y in (-.5,shape[1]-.5):
            for x in (-.5,shape[0]-.5):
                point=np.asarray([x,y,z,1.]);errors.append(float(np.max(np.abs((a@point)[:3]-(b@point)[:3]))))
    maximum=max(errors)
    if maximum>1e-6:raise AssertionError('World-coordinate outer-grid corner error exceeds 1e-6 mm.')
    return dict(shape_xyz=actual['shape'],unit=actual['unit'],maximum_outer_corner_error_mm=maximum,
        affine_ras_mm=actual['affine'])


def native_sparse(matrix):
    rows,columns,counts=matrix['row_values'],matrix['column_values'],matrix['counts']
    assert len(counts)==len(rows)*len(columns)
    return dict(row_values=rows,column_values=columns,entries=[[row,column,counts[i*len(columns)+j]]
        for i,row in enumerate(rows) for j,column in enumerate(columns) if counts[i*len(columns)+j]!=0])


def check_probe(probe,reference,implementation):
    if probe['implementation_sha256']!=implementation:raise AssertionError('Matrix probe implementation differs from wheel/CLI snapshot.')
    result=dict(source_geometry=check_geometry(probe['source_geometry'],reference['geometry']),
        cropped_geometry=check_geometry(probe['crop_geometry'],reference['matrix_case']['geometry']))
    for native,key in [('source_image_f64_le_sha256','image_f64_le_sha256'),('source_mask_u32_le_sha256','mask_u32_le_sha256')]:
        if probe[native]!=reference[key]:raise AssertionError('Native full source buffer digest differs: '+native)
    for native,key in [('crop_image_f64_le_sha256','image_f64_le_sha256'),('crop_mask_u32_le_sha256','mask_u32_le_sha256'),('discrete_u32_le_sha256','discrete_u32_le_sha256')]:
        if probe[native]!=reference['matrix_case'][key]:raise AssertionError('Native cropped/discrete buffer digest differs: '+native)
    matrices=probe['matrices'];expected=reference['matrix_case'];comparisons=0
    for family in ('glcm','glrlm'):
        if [row['offset'] for row in matrices[family]]!=[row['offset_xyz'] for row in expected[family]]:
            raise AssertionError('Directional offset identity/order differs: '+family)
        for actual,target in zip(matrices[family],expected[family]):
            if native_sparse(actual['matrix'])!=target['matrix']:raise AssertionError('Exact sparse directional counts differ: '+family)
            comparisons+=1
    for family in ('glszm','gldm'):
        if native_sparse(matrices[family])!=expected[family]:raise AssertionError('Exact sparse counts differ: '+family)
        comparisons+=1
    if [(v['gray_level'],v['count']) for v in matrices['ngtdm']]!=[(v['gray_level'],v['count']) for v in expected['ngtdm']]:
        raise AssertionError('NGTDM gray labels/counts differ.')
    sums=[numeric(a['difference_sum'],b['difference_sum'],'ngtdm_difference_sum','fixed_block','rust_core')
        for a,b in zip(matrices['ngtdm'],expected['ngtdm'])]
    if not all(row['passed'] for row in sums):raise AssertionError('NGTDM difference sums exceed frozen tolerance.')
    result.update(exact_count_matrices=comparisons,ngtdm_rows=len(sums),ngtdm_difference_sums=sums,
        full_and_cropped_buffer_digests='exact',discrete_gray_digest='exact',passed=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--cli',type=Path,required=True)
    parser.add_argument('--build-manifest',type=Path,required=True)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if output.exists():raise FileExistsError('Native acceptance requires a fresh artifact output.')
    if any((parent/'.git').exists() for parent in [output,*output.parents]):raise ValueError('CT raw/native results stay outside Git worktrees.')
    protocol=json.loads((PROFILE/'protocol.json').read_text(encoding='utf-8'))
    reference=json.loads((PROFILE/'reference.json').read_text(encoding='utf-8'))
    crop=json.loads((PROFILE/'bbox_crop_reference.json').read_text(encoding='utf-8'))
    build=json.loads(args.build_manifest.read_text(encoding='utf-8'));implementation=build['core_implementation_sha256']
    if sha(args.cli.read_bytes())!=build['cli']['sha256']:raise AssertionError('CLI binary differs from immutable build manifest.')
    for name,record in protocol['source']['files'].items():
        raw=(args.data_dir/name).read_bytes()
        if len(raw)!=record['bytes'] or sha(raw)!=record['sha256']:raise AssertionError('External CT/ROI source bytes differ.')
    sys.meta_path.insert(0,RejectReference())
    import numpy as np
    import pandas as pd
    import ov_radiomics as rad
    package=Path(rad.__file__).parent.parent
    checked=0
    for name,expected in build['wheel_files_sha256'].items():
        if name.startswith('ov_radiomics/'):
            if sha((package/name).read_bytes())!=expected:raise AssertionError('Installed wheel file differs from immutable snapshot: '+name)
            checked+=1
    output.mkdir(parents=True)
    receipt=dict(schema='radiomics.public_ct_v1.native_acceptance.v1',status='running',license=protocol['license'],
        protocol_sha256=sha((PROFILE/'protocol.json').read_bytes()),reference_sha256=sha((PROFILE/'reference.json').read_bytes()),
        bbox_reference_sha256=sha((PROFILE/'bbox_crop_reference.json').read_bytes()),
        source_revision=build['source_revision'],core_implementation_sha256=implementation,
        build_manifest_sha256=sha(args.build_manifest.read_bytes()),installed_package=str(package/'ov_radiomics'),
        installed_snapshot_files_verified=checked,cli_sha256=build['cli']['sha256'],python=sys.version,
        numpy=np.__version__,pandas=pd.__version__,comparisons=[],geometry={},commands=[],
        performance_measured=False,ibsi_ct_conformance_claim=False)
    try:
        image=rad.image.read_nifti(args.data_dir/'image.nii.gz');mask=rad.mask.read_nifti(args.data_dir/'mask.nii.gz')
        receipt['geometry']['python_image']=check_geometry(image.geometry,reference['geometry'])
        receipt['geometry']['python_mask']=check_geometry(mask.geometry,reference['geometry'])
        if sha(image.array.astype('<f8').tobytes(order='C'))!=reference['image_f64_le_sha256']:raise AssertionError('Python/native full-image buffer differs.')
        if sha(mask.array.astype('<u4').tobytes(order='C'))!=reference['mask_u32_le_sha256']:raise AssertionError('Python/native source mask differs.')
        bounds=rad.mask.resegment(image,mask,lower=-500.,upper=400.)
        if sha(bounds.array.astype('<u4').tobytes(order='C'))!=reference['cases']['PAT1_resegmented_3d_bin25']['intensity_mask_u32_le_sha256']:
            raise AssertionError('Native intensity resegmentation mask positions differ.')
        native_config=dict(profile='python_golden_v1',discretization=dict(method='fixed_width',width=25.),
            aggregation='direction_mean',gldm_alpha=0,voxel_array_shift=0.)
        cases=[(case['id'],reference['cases'][case['id']]['features']|reference['basic_shape'],
            [] if case['intensity_range'] is None else [dict(operation='resegment',lower=-500.,upper=400.)]) for case in protocol['cases']]
        cases.append(('PAT1_tight_bbox_3d_bin25',crop['features']|crop['basic_shape'],[dict(operation='crop_voxels',margin_voxels=[0,0,0])]))
        env=dict(os.environ,PATH='')
        for name,expected,preparation in cases:
            folder=output/name;folder.mkdir()
            config=dict(extraction=native_config,labels=[1],preparation=preparation)
            samples=pd.DataFrame([dict(ID=name,patient_id='PAT1',modality='CT',region='GTV-1',image=str(args.data_dir/'image.nii.gz'),mask=str(args.data_dir/'mask.nii.gz'))])
            result=rad.extract(samples,params=config,backend='rust',profile='python_golden_v1',n_jobs=1,cache=None)
            if result.config['implementation_sha256']!=implementation:raise AssertionError('Python extraction core differs from immutable snapshot.')
            if result.config['status']!='complete' or result.config['failures']:raise AssertionError('Native batch lost/failed CT sample.')
            if list(result.features.columns)!=['ID',*protocol['output_feature_order']]:raise AssertionError('Native feature order differs from frozen export order.')
            if result.feature_table.metadata.patient_id.tolist()!=['PAT1']:raise AssertionError('Patient identity was lost.')
            write_json(folder/'api-extraction.json',result.config)
            api=result.features.iloc[0].to_dict()
            for feature,value in expected.items():receipt['comparisons'].append(numeric(api[feature],value,feature,name,'python_rust_api'))
            cohort=folder/'cohort.csv'
            with cohort.open('w',newline='',encoding='utf-8') as file:
                writer=csv.DictWriter(file,fieldnames=['sample_id','patient_id','modality','region','image','mask']);writer.writeheader()
                writer.writerow(dict(sample_id=name,patient_id='PAT1',modality='CT',region='GTV-1',image=str(args.data_dir/'image.nii.gz'),mask=str(args.data_dir/'mask.nii.gz')))
            write_json(folder/'config.json',config)
            argv=[str(args.cli.resolve()),'extract','--cohort',str(cohort),'--config',str(folder/'config.json'),'--output',str(folder/'cli'),'--n-jobs','1']
            completed=subprocess.run(argv,env=env,capture_output=True)
            (folder/'cli.stdout').write_bytes(completed.stdout);(folder/'cli.stderr').write_bytes(completed.stderr)
            receipt['commands'].append(dict(argv=argv,exit_code=completed.returncode,python_path='empty PATH',stderr_sha256=sha(completed.stderr)))
            if completed.returncode:raise AssertionError('Python-free CLI extraction failed.')
            batch=json.loads((folder/'cli/extraction.json').read_text(encoding='utf-8'))
            if batch['implementation_sha256']!=implementation or batch['status']!='complete' or batch['failures']:
                raise AssertionError('CLI implementation/status differs.')
            with (folder/'cli/features.csv').open(newline='',encoding='utf-8') as file:
                reader=csv.DictReader(file);rows=list(reader)
                if reader.fieldnames!=['sample_id',*protocol['output_feature_order']]:raise AssertionError('CLI feature order differs.')
            if len(rows)!=1 or rows[0]['sample_id']!=name:raise AssertionError('CLI sample ID differs.')
            for feature,value in expected.items():
                receipt['comparisons'].append(numeric(rows[0][feature],value,feature,name,'rust_cli'))
                if float(rows[0][feature])!=float(api[feature]):raise AssertionError('CLI/API scalar differs for identical core.')
        probe=json.loads(args.probe.read_text(encoding='utf-8'))
        receipt['matrix']=check_probe(probe,reference,implementation)
        receipt['probe_sha256']=sha(args.probe.read_bytes())
        receipt['forbidden_modules_loaded']=sorted({name.split('.')[0] for name in sys.modules}&FORBIDDEN)
        if receipt['forbidden_modules_loaded']:raise AssertionError('Optional Python/reference backend was loaded.')
        receipt['passed_comparisons']=sum(row['passed'] for row in receipt['comparisons'])
        receipt['total_comparisons']=len(receipt['comparisons'])
        if receipt['passed_comparisons']!=receipt['total_comparisons']:raise AssertionError('Frozen numerical comparisons failed; inspect raw per-feature errors.')
        receipt['status']='passed'
    except Exception as error:
        receipt.update(status='failed',error=f'{type(error).__name__}: {error}')
    write_json(output/'acceptance.json',receipt)
    print(json.dumps({key:receipt.get(key) for key in ('status','error','passed_comparisons','total_comparisons','core_implementation_sha256')},indent=2))
    return 0 if receipt['status']=='passed' else 1


if __name__=='__main__':sys.exit(main())
