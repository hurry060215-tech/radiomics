"""Predeclared Radiomics Benchmark v2; reference/native work in isolated workers.

--prepare creates deterministic inputs only. --run requires the main-reviewed
plan SHA, an immutable native wheel and its build manifest. It never records or
changes Golden expected values, tolerances, seeds or the performance case grid.
"""
import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import time
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[1]
from benchmark_protocol import latency_summary, machine_manifest, paired_summary, sha256_file


def clean(value):
    if isinstance(value,dict):
        return {str(key):clean(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [clean(item) for item in value]
    if hasattr(value,'tolist'):
        return clean(value.tolist())
    if isinstance(value,float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else 'Inf' if value>0 else '-Inf'
    return value


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(clean(value),handle,allow_nan=False,ensure_ascii=False,sort_keys=True,indent=2)
        handle.write('\n')


def json_hash(value):
    return hashlib.sha256(json.dumps(clean(value),allow_nan=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def load_plan(path,approved=None):
    raw=Path(path).read_bytes();plan=json.loads(raw)
    digest=hashlib.sha256(raw).hexdigest()
    if approved is not None and digest!=approved:
        raise ValueError('Plan bytes differ from the explicitly approved SHA256; do not measure an unreviewed amendment.')
    if plan['schema']!='radiomics.benchmark.v2.radiomics_plan.v1' or plan['suite']!='radiomics':
        raise ValueError('Expected the predeclared Radiomics-only plan.')
    if sum(map(len,plan['reference']['features'].values()))!=55 or plan['execution']['paired_repetitions']!=7:
        raise ValueError('The declared 55-feature/seven-pair scope is required.')
    return plan,digest


def load_ct_plan(path,base_sha,approved=None):
    raw=Path(path).read_bytes();supplement=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
    if approved is not None and digest!=approved:
        raise ValueError('CT supplemental bytes differ from their independently reviewed SHA256.')
    if supplement['schema']!='radiomics.benchmark.v2.public_ct_append.v1' or supplement['base_plan_sha256']!=base_sha:
        raise ValueError('CT supplement must append to the exact unchanged base plan.')
    if supplement['original_case_count_unchanged']!=58 or supplement['total_primary_cases']!=60 or len(supplement['cases'])!=2:
        raise ValueError('The CT addition may only append the two predeclared cases to the original58.')
    if any(case['roi_count']!=1 or case['threads']!=1 for case in supplement['cases']):
        raise ValueError('CT calls must retain one ROI and one outer worker.')
    return supplement,digest


def load_dimension_binding(path,base_sha,ct_sha,approved):
    raw=Path(path).read_bytes();binding=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
    if digest!=approved or binding['base_plan_sha256']!=base_sha or ct_sha is not None and binding['ct_append_sha256']!=ct_sha:
        raise ValueError('Inactive-axis binding is absent/unapproved or points to different immutable plans.')
    if binding['schema']!='radiomics.benchmark.v2.inactive_axis_binding.v1' or binding['effective_native_and_reference_vector']!=[1,1,0]:
        raise ValueError('The only approved interface clarification keeps xy1 and sets inactive2D z to0.')
    return binding,digest


def bind_inactive_axis(declared,binding):
    # Case identity/data/order remain intact. Do not silently infer3D versus2D
    # from a singleton shape axis; the binding names the declared true2D case.
    result=[];changed=[];scope=binding['applies_only_to']
    for original in declared:
        case=dict(original)
        if case['dataset']==scope['dataset'] and case['level']==scope['level'] and case['state'] in scope['states']:
            steps=[]
            for step in case['preparation']:
                step=dict(step)
                if step['operation']==scope['operation']:
                    if step['margin_voxels']!=binding['original_base_vector']:
                        raise ValueError('Inactive-axis binding does not match the preserved original configuration.')
                    step['margin_voxels']=binding['effective_native_and_reference_vector']
                steps.append(step)
            case['preparation']=steps
            case['configuration_sha256']=json_hash({key:value for key,value in case.items() if key not in ('id','configuration_sha256')})
            changed.append(case['id'])
        result.append(case)
    if len(changed)!=2:
        raise ValueError('Exactly the two approved true2D L3 states must be bound; no other case changes are allowed.')
    return result


def read_nifti_fixture(path):
    """Fixture preparation only: scalar NIfTI-1 with explicit sform, not runtime reader replacement."""
    import numpy as np
    raw=Path(path).read_bytes()
    if str(path).endswith('.gz'):
        raw=gzip.decompress(raw)
    if struct.unpack_from('<i',raw)[0]!=348:
        raise ValueError('Benchmark fixtures require explicit little-endian NIfTI-1.')
    dims=struct.unpack_from('<8h',raw,40);ndim=dims[0]
    if ndim not in (2,3) or struct.unpack_from('<h',raw,254)[0]<=0:
        raise ValueError('Benchmark fixtures require 2D/3D and an explicit sform.')
    code=struct.unpack_from('<h',raw,70)[0]
    dtype={2:'u1',4:'<i2',8:'<i4',16:'<f4',64:'<f8',256:'i1',512:'<u2',768:'<u4',1024:'<i8',1280:'<u8'}[code]
    shape=tuple(dims[1:ndim+1]);offset=int(struct.unpack_from('<f',raw,108)[0])
    array=np.frombuffer(raw,dtype=dtype,count=math.prod(shape),offset=offset).reshape(tuple(reversed(shape))).astype(np.float64)
    slope,intercept=struct.unpack_from('<ff',raw,112)
    if slope!=0:
        array=array*slope+intercept
    affine=np.eye(4);affine[:3]=np.asarray(struct.unpack_from('<12f',raw,280)).reshape(3,4)
    if raw[123]&7 !=2:
        raise ValueError('These declared benchmark fixtures must carry millimeter units.')
    return array,affine


def write_nifti_fixture(path,array,affine,is_mask=False):
    import numpy as np
    array=np.asarray(array,dtype=np.uint8 if is_mask else '<f8')
    header=bytearray(352);struct.pack_into('<i',header,0,348)
    shape=list(reversed(array.shape));struct.pack_into('<8h',header,40,array.ndim,*shape,*([1]*(7-len(shape))))
    struct.pack_into('<hh',header,70,2 if is_mask else 64,8 if is_mask else 64)
    spacing=[float(np.linalg.norm(np.asarray(affine)[:3,i])) for i in range(3)]
    struct.pack_into('<8f',header,76,1,*spacing,1,1,1,1)
    struct.pack_into('<ff',header,108,352,1);header[123]=2
    struct.pack_into('<hh',header,252,0,1);struct.pack_into('<12f',header,280,*np.asarray(affine)[:3].ravel())
    header[344:348]=b'n+1\0'
    Path(path).write_bytes(gzip.compress(bytes(header)+array.tobytes(order='C'),mtime=0))


def synthetic_dataset(spec,generator):
    import numpy as np
    shape=spec['shape_xyz'];axes=np.indices(tuple(reversed(shape)),dtype=np.uint64)
    x=axes[-1];y=axes[-2];z=axes[-3] if len(shape)==3 else np.zeros_like(x)
    values=1+((13*x+17*y+29*z+generator['seed'])^((x*y+3*z+7*x*z)%251))%spec['gray_levels']
    distance=np.zeros_like(x,dtype=float)
    for axis,size in zip(reversed(axes),shape):
        distance+=((2*axis.astype(float)-(size-1))/(size-3))**2
    mask=(distance<=1).astype(np.uint8)
    affine=np.eye(4);affine[:3,:3]=np.diag(generator['spacing_xyz_mm']);affine[:3,3]=generator['origin_ras_mm']
    return values.astype(np.float64),mask,affine


def cohort_file(path,dataset,count):
    with Path(path).open('x',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle);writer.writerow(['sample_id','patient_id','image','mask'])
        for i in range(count):
            writer.writerow([f"{dataset}-R{i:04}",f'P{i:04}','image.nii.gz','mask.nii.gz'])


def prepare(plan,plan_sha,output,ct_plan=None,ct_sha=None,ct_source=None,ct_reference_paths=None):
    import numpy as np
    output=Path(output)
    if output.exists():
        raise FileExistsError('Benchmark fixture output must be a fresh directory.')
    output.mkdir(parents=True)
    records=[]
    for spec in plan['datasets']:
        directory=output/spec['id'];directory.mkdir()
        source=None
        if 'source' in spec:
            source=ROOT/spec['source'];frozen=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
            for name in ('image.nii.gz','mask.nii.gz'):
                if sha256_file(source/name)!=frozen['files_sha256'][name]:
                    raise ValueError('IBSI source bytes no longer match their independent frozen manifest.')
                shutil.copyfile(source/name,directory/name)
            for name in ('LICENSE.md','SOURCE_README.md'):
                shutil.copyfile(source/name,directory/name)
            values,affine=read_nifti_fixture(directory/'image.nii.gz');mask,mask_affine=read_nifti_fixture(directory/'mask.nii.gz')
            if values.shape!=mask.shape or not np.array_equal(affine,mask_affine) or not np.isin(mask,[0,1]).all():
                raise ValueError('IBSI fixture geometry/label contract differs.')
            mask=mask.astype(np.uint8)
        else:
            values,mask,affine=synthetic_dataset(spec,plan['synthetic_generator'])
            write_nifti_fixture(directory/'image.nii.gz',values,affine);write_nifti_fixture(directory/'mask.nii.gz',mask,affine,True)
        if list(reversed(values.shape))!=spec['shape_xyz']:
            raise ValueError('Prepared fixture shape differs from the predeclared grid.')
        np.save(directory/'values.npy',values,allow_pickle=False);np.save(directory/'mask.npy',mask,allow_pickle=False)
        counts=set()
        for level in spec['levels']:
            declaration=plan['levels'][level]
            counts.update(declaration.get('roi_counts',[declaration.get('roi_count_per_call',1)]))
        for count in sorted(counts):
            cohort_file(directory/f'cohort-{count}.csv',spec['id'],count)
        record=dict(id=spec['id'],directory=spec['id'],shape_xyz=spec['shape_xyz'],affine_ras_mm=affine.tolist(),
            roi_voxels=int(mask.sum()),present_gray_values=np.unique(values[mask==1]).tolist(),
            roi_count_files=sorted(counts),license=spec.get('license',plan['synthetic_generator']['license']),
            original_source=None if source is None else str(source),
            files_sha256={p.name:sha256_file(p) for p in sorted(directory.iterdir())})
        record['input_sha256']=json_hash(record['files_sha256']);records.append(record)
    if ct_plan is not None:
        records.append(prepare_ct(plan,ct_plan,ct_sha,output,ct_source,ct_reference_paths))
    manifest=dict(schema='radiomics.benchmark.v2.inputs.v1',approved_plan_sha256=plan_sha,
        synthetic_generator=plan['synthetic_generator'],datasets=records,
        repeated_roi_disclosure='Each batch repeats the same fixed image/ROI with distinct sample/patient IDs; uncached calls do not reuse computed features.',
        ibsi_attribution='IBSI-1 digital phantom, Alex Zwanenburg, CC-BY-4.0 https://creativecommons.org/licenses/by/4.0/ ; original NIfTI/license/source README bytes copied unchanged; no new conformance claim.')
    if ct_plan is not None:
        manifest.update(ct_supplement_sha256=ct_sha,ct_license=ct_plan['dataset']['license'],ct_raw_and_prepared_data_external_only=True)
    write_json(output/'manifest.json',manifest)
    return manifest


def prepare_ct(plan,supplement,supplement_sha,output,source,reference_paths):
    """External licensed raw/cropped data only; no native or feature extraction."""
    import numpy as np
    output=Path(output).resolve()
    if output==ROOT or ROOT in output.parents:
        raise ValueError('Noncommercial CT raw/prepared arrays must stay in external artifacts, outside the source checkout.')
    if source is None:
        raise ValueError('The approved public CT external source directory is required.')
    source=Path(source);spec=supplement['dataset'];directory=output/spec['id'];directory.mkdir()
    frozen={}
    for key,binding in supplement['reference_bindings'].items():
        path=Path((reference_paths or {}).get(key) or ROOT/binding['repository_path'])
        if sha256_file(path)!=binding['sha256']:
            raise ValueError('CT LF reference/profile differs from the pre-native approved binding: '+key)
        frozen[key]=json.loads(path.read_text(encoding='utf-8'))
        shutil.copyfile(path,directory/Path(binding['repository_path']).name)
    if frozen['protocol']['features']!=plan['reference']['features']:
        raise ValueError('CT and base plan must use the exact same55 selected feature names.')
    crop_reference=frozen['tight_bbox_reference']
    if crop_reference['protocol_sha256']!=supplement['reference_bindings']['protocol']['sha256'] or crop_reference['full_reference_sha256']!=supplement['reference_bindings']['original_reference']['sha256']:
        raise ValueError('The approved tight-crop reference is not bound to the original public CT reference.')
    for name,digest in spec['source_files_sha256'].items():
        if sha256_file(source/name)!=digest:
            raise ValueError('Public CT source/license bytes differ: '+name)
        shutil.copyfile(source/name,directory/name)
    values,affine=read_nifti_fixture(directory/'image.nii.gz');mask,mask_affine=read_nifti_fixture(directory/'mask.nii.gz')
    if list(reversed(values.shape))!=spec['source_shape_xyz'] or not np.array_equal(affine,spec['source_affine_ras_mm']) or not np.array_equal(affine,mask_affine):
        raise ValueError('Public CT source geometry differs from the frozen complete affine/grid.')
    if not np.isin(mask,[0,1]).all() or int(np.count_nonzero(mask==1))!=spec['morphology_and_intensity_roi_voxels']:
        raise ValueError('CT morphology/intensity ROI is not the125256-voxel original mask.')
    start,size=spec['tight_crop']['start_xyz'],spec['tight_crop']['size_xyz'];slices=tuple(slice(a,a+b) for a,b in zip(reversed(start),reversed(size)))
    values=np.ascontiguousarray(values[slices],dtype='<f8');mask=np.ascontiguousarray(mask[slices],dtype=np.uint8)
    prepared_affine=affine.copy();prepared_affine[:3,3]+=prepared_affine[:3,:3]@np.asarray(start)
    if not np.array_equal(prepared_affine,spec['prepared_affine_ras_mm']) or not np.array_equal(prepared_affine,crop_reference['geometry']['affine_ras_mm']):
        raise ValueError('CT prepared affine differs from the independent frozen tight-crop geometry.')
    if hashlib.sha256(values.tobytes()).hexdigest()!=crop_reference['image_f64_le_sha256'] or hashlib.sha256(mask.astype('<u4').tobytes()).hexdigest()!=crop_reference['mask_u32_le_sha256']:
        raise ValueError('Prepared CT arrays differ from their independently frozen byte-level reference.')
    np.save(directory/'values.npy',values,allow_pickle=False);np.save(directory/'mask.npy',mask,allow_pickle=False)
    cohort_file(directory/'cohort-1.csv',spec['id'],1)
    record=dict(id=spec['id'],directory=spec['id'],shape_xyz=size,affine_ras_mm=prepared_affine.tolist(),
        source_shape_xyz=spec['source_shape_xyz'],source_affine_ras_mm=affine.tolist(),roi_voxels=int(mask.sum()),
        present_gray_values=np.unique(values[mask==1]).tolist(),roi_count_files=[1],license=spec['license'],
        original_source=str(source.resolve()),ct_supplement_sha256=supplement_sha,reference_bindings=supplement['reference_bindings'],
        files_sha256={path.name:sha256_file(path) for path in sorted(directory.iterdir())})
    record['input_sha256']=json_hash(record['files_sha256'])
    return record


def expected_columns(plan,ndim,family=None):
    names=[]
    for key,features in plan['reference']['features'].items():
        if family is None or family==key:
            names.extend('original_'+key+'_'+name for name in features)
    if family in (None,'shape_basic'):
        names+=['original_shape_basic_VoxelCount']
        for axis in 'XYZ':
            names+=['original_shape_basic_BoundingBoxMin'+axis+'_index','original_shape_basic_BoundingBoxSize'+axis+'_voxels','original_shape_basic_Centroid'+axis+'_mm']
        names+=['original_shape_basic_Volume_mm3' if ndim==3 else 'original_shape_basic_Area_mm2']
    return sorted(names)


def shape_basic_spec(mask,affine):
    """Separate project descriptor specification; no IBSI mesh morphology claim."""
    import numpy as np
    coordinates=np.argwhere(mask==1)[:,::-1]
    if mask.ndim==2:
        coordinates=np.column_stack((coordinates,np.zeros(len(coordinates),dtype=int)))
    minimum,maximum=coordinates.min(axis=0),coordinates.max(axis=0)
    centroid=np.asarray(affine)[:3,:3]@coordinates.mean(axis=0)+np.asarray(affine)[:3,3]
    values={'original_shape_basic_VoxelCount':float(len(coordinates))}
    for i,axis in enumerate('XYZ'):
        values['original_shape_basic_BoundingBoxMin'+axis+'_index']=float(minimum[i])
        values['original_shape_basic_BoundingBoxSize'+axis+'_voxels']=float(maximum[i]-minimum[i]+1)
        values['original_shape_basic_Centroid'+axis+'_mm']=float(centroid[i])
    if mask.ndim==3:
        values['original_shape_basic_Volume_mm3']=float(len(coordinates)*abs(np.linalg.det(np.asarray(affine)[:3,:3])))
    else:
        values['original_shape_basic_Area_mm2']=float(len(coordinates)*np.linalg.norm(np.cross(np.asarray(affine)[:3,0],np.asarray(affine)[:3,1])))
    return values


def compare_artifacts(reference,native,plan):
    a=json.loads(Path(reference).read_text(encoding='utf-8'));b=json.loads(Path(native).read_text(encoding='utf-8'))
    if a['sample_ids']!=b['sample_ids'] or a['feature_names']!=b['feature_names']:
        return dict(passed=False,reason='Exact sample ID/order or feature order mismatch.')
    if len(a['values'])!=len(b['values']):
        return dict(passed=False,reason='Feature row counts differ.')
    failures=[];maximum_absolute_error=0.
    atol,rtol=plan['correctness']['atol'],plan['correctness']['rtol']
    for row,(ar,br) in enumerate(zip(a['values'],b['values'])):
        if len(ar)!=len(br) or len(ar)!=len(a['feature_names']):
            return dict(passed=False,reason='Feature widths differ.')
        for col,(expected,actual) in enumerate(zip(ar,br)):
            name=a['feature_names'][col]
            if isinstance(expected,str) or isinstance(actual,str):
                passed=actual==expected
            else:
                error=abs(actual-expected);maximum_absolute_error=max(maximum_absolute_error,error)
                if '_shape_basic_BoundingBox' in name or name.endswith('_VoxelCount'):
                    passed=actual==expected
                elif '_shape_basic_Centroid' in name:
                    passed=error<=plan['shape_basic']['geometry_tolerance_mm']
                else:
                    passed=error<=atol+rtol*abs(expected)
            if not passed:
                failures.append(dict(sample_id=a['sample_ids'][row],feature=name,reference=expected,actual=actual))
    return dict(passed=not failures,compared_values=sum(map(len,a['values'])),maximum_absolute_error=maximum_absolute_error,
        mismatch_count=len(failures),first_mismatches=failures[:20])


def safe_compare_artifacts(reference,native,plan):
    try:
        return compare_artifacts(reference,native,plan)
    except Exception as exc:
        return dict(passed=False,reason='Numerical comparison artifact/protocol failed; completed backend observations are retained.',
            error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())


def peak_resident_bytes():
    if sys.platform=='win32':
        return process_memory(os.getpid())[1]
    import resource
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform=='darwin' else peak*1024)


def process_memory(pid):
    """Current/peak RSS for an owned worker; no process-wide settings or psutil dependency."""
    if sys.platform=='win32':
        import ctypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',ctypes.c_ulong),('page_faults',ctypes.c_ulong)]+[(name,ctypes.c_size_t) for name in (
                'peak_working_set','working_set','peak_paged_pool','paged_pool','peak_nonpaged_pool','nonpaged_pool','pagefile','peak_pagefile')]
        kernel=ctypes.windll.kernel32;psapi=ctypes.windll.psapi
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
        handle=kernel.OpenProcess(0x0400|0x0010,False,pid)
        if not handle:
            return None,None
        try:
            counter=Counters();counter.cb=ctypes.sizeof(counter)
            if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(counter),ctypes.sizeof(counter)):
                return None,None
            return int(counter.working_set),int(counter.peak_working_set)
        finally:
            kernel.CloseHandle(handle)
    try:
        lines=Path(f'/proc/{pid}/status').read_text(encoding='utf-8').splitlines()
        stats={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in lines if ':' in line}
        return int(stats['VmRSS'].split()[0])*1024,int(stats['VmHWM'].split()[0])*1024
    except (OSError,ValueError,KeyError):
        return None,None


def package_files(package):
    directory=Path(package.__file__).resolve().parent
    return {str(p.relative_to(directory)).replace('\\','/'):sha256_file(p) for p in sorted(directory.rglob('*'))
        if p.is_file() and p.suffix.lower() in ('.py','.pyd','.so','.dll')}


def validate_reference_files(actual,expected,revision,declared_revision):
    if not expected or revision!=declared_revision:
        raise RuntimeError('Reference source/build provenance is missing or has a different approved public revision; pending, not verified.')
    if not any(name.endswith(('.pyd','.so')) for name in expected) or not any(name.endswith('.py') for name in expected):
        raise RuntimeError('Expected reference provenance must include source and compiled-extension hashes.')
    for name,digest in expected.items():
        if actual.get(name)!=digest:
            raise RuntimeError(f'Reference file {name} differs from the frozen public3.0.1 source/build artifact. A different platform build needs independently reviewed expected build hashes, not a version-string pass.')


def worker_setup(request,plan):
    import numpy as np
    import pandas as pd
    audit=bool(request.get('describe_only'))
    versions=request.get('preflight_environment',{}).get('versions',{})
    if audit:
        versions={}
        for name in ('numpy','pandas','matplotlib','SimpleITK','pyradiomics','scipy','scikit-learn','statsmodels','joblib','threadpoolctl'):
            try:
                versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name]=None
    extraction=request.get('case',{}).get('native_extraction',plan['native_extraction'])
    settings=dict(plan['reference']['settings'])
    if extraction['discretization']['method']!='fixed_width':
        raise RuntimeError('This approved benchmark declares fixed widths, not an unreviewed alternative binning setup.')
    shift=extraction['voxel_array_shift']
    # PyRadiomics's public parameter schema types this as int; the approved
    # benchmark shift is exactly zero, so preserve its value without truncation.
    shift=int(shift) if float(shift).is_integer() else shift
    settings.update(binWidth=extraction['discretization']['width'],gldm_a=extraction['gldm_alpha'],voxelArrayShift=shift)
    state=dict(request=request,plan=plan,np=np,pd=pd,native_extraction=extraction,reference_settings=settings)
    if request['backend']=='rust':
        import ov_radiomics as rad
        from ov_radiomics import _native
        root=Path(rad.__file__).resolve()
        if 'site-packages' not in root.parts:
            raise RuntimeError('Native benchmark must use a dedicated installed immutable wheel, never an editable source tree.')
        files=None
        if audit:
            files=package_files(rad)
            if files!=request['expected_native_package_files']:
                raise RuntimeError('Installed native package bytes differ from the frozen wheel.')
        elif str(root.parent)!=request['expected_package_path']:
            raise RuntimeError('Measured native module came from a different installed package path.')
        if audit:
            import inspect
            if 'families' not in inspect.signature(rad.ImageSample.extract).parameters:
                raise RuntimeError('The frozen wheel lacks the reviewed selective-family interface; L1 must not time full extraction.')
        state['rad']=rad
        build_info=None
        if audit and hasattr(_native,'build_info'):
            build_info=_native.build_info()
            if isinstance(build_info,str):
                build_info=json.loads(build_info)
        details=dict(package_path=str(root.parent),core_version=_native.CORE_VERSION,native_build_info=build_info)
        if not hasattr(_native,'IMPLEMENTATION_SHA256') or _native.IMPLEMENTATION_SHA256!=request['native_build']['core_implementation_sha256']:
            raise RuntimeError('Installed native core implementation constant differs from the mandatory immutable build receipt.')
        state['implementation_sha256']=_native.IMPLEMENTATION_SHA256
        details['core_implementation_sha256']=_native.IMPLEMENTATION_SHA256
        if audit:
            details['package_files_sha256']=files
        if any(name in sys.modules for name in ('radiomics','SimpleITK','sklearn','statsmodels')):
            raise RuntimeError('Native measurement unexpectedly imported an external reference backend.')
    else:
        import logging
        import radiomics
        import SimpleITK as sitk
        from radiomics.featureextractor import RadiomicsFeatureExtractor
        if radiomics.__version__!='v3.0.1' or np.__version__!=plan['reference']['numpy'] or sitk.Version_VersionString()!=plan['reference']['simpleitk']:
            raise RuntimeError('Reference must actually import frozen public PyRadiomics3.0.1/NumPy1.23.0/SimpleITK2.4.1; no 3.1 substitution.')
        if any(name.startswith('ov_radiomics') for name in sys.modules):
            raise RuntimeError('Reference worker imported the native project package.')
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        radiomics.setVerbosity(logging.ERROR)
        classes={}
        for family in plan['reference']['features']:
            module=importlib.import_module('radiomics.'+family)
            classes[family]=getattr(module,'RadiomicsFirstOrder' if family=='firstorder' else 'Radiomics'+family.upper())
        extractor=RadiomicsFeatureExtractor(dict(imageType={'Original':{}},featureClass=plan['reference']['features'],setting=settings))
        if set(extractor.enabledImagetypes)!= {'Original'} or extractor.enabledFeatures!=plan['reference']['features']:
            raise RuntimeError('Reference enabled feature/image scope differs from the frozen explicit selection.')
        state.update(sitk=sitk,reference_classes=classes,extractor=extractor)
        package_path=str(Path(radiomics.__file__).resolve().parent)
        if not audit and package_path!=request['expected_package_path']:
            raise RuntimeError('Measured reference module came from a different verified package path.')
        details=dict(package_path=package_path,
            pyradiomics_version=radiomics.__version__,effective_extractor_settings=extractor.settings,
            enabled_features=extractor.enabledFeatures,enabled_image_types=extractor.enabledImagetypes,
            simpleitk_effective_threads=sitk.ProcessObject.GetGlobalDefaultNumberOfThreads())
        if audit:
            files=package_files(radiomics)
            expected=request.get('expected_reference',{})
            validate_reference_files(files,expected.get('files'),expected.get('revision'),plan['reference']['pyradiomics_revision'])
            details.update(package_files_sha256=files,approved_public_revision=expected['revision'],reference_build_hashes_verified=True)
    import threadpoolctl
    # The limiter is local to the owned worker and restored by worker() below.
    # Observe the effective libraries after applying the declared inner limit.
    state['thread_limiter']=threadpoolctl.threadpool_limits(limits=1)
    pools=threadpoolctl.threadpool_info()
    validate_thread_resources(pools,details.get('simpleitk_effective_threads'),request['backend'])
    state['observed_thread_libraries']=pools
    if audit:
        dependencies={}
        for name in ('numpy.core._multiarray_umath','SimpleITK._SimpleITK'):
            module=sys.modules.get(name);path=getattr(module,'__file__',None)
            if path is not None:
                dependencies[str(Path(path).resolve())]=sha256_file(path)
        for pool in pools:
            path=pool.get('filepath')
            if path and Path(path).is_file():
                dependencies[str(Path(path).resolve())]=sha256_file(path)
        details.update(python_executable=sys.executable,python_version=platform.python_version(),versions=versions,
            thread_environment={key:os.environ.get(key) for key in plan['execution']['thread_environment']},observed_thread_libraries=pools,
            dependency_binary_files_sha256=dependencies,
            resident_memory_metric='Windows PeakWorkingSetSize' if sys.platform=='win32' else 'Linux ru_maxrss*1024',
            peak_before_fixture_bytes=peak_resident_bytes())
    state['environment']=details
    return state


def validate_thread_resources(pools,itk_threads,backend):
    recognized=[pool for pool in pools or [] if pool.get('user_api') in ('blas','openmp')]
    if not recognized or any(pool.get('num_threads')!=1 for pool in recognized):
        raise RuntimeError('Actual BLAS/OpenMP thread resources are unavailable or exceed the predeclared inner limit1; not a matched-resource comparison.')
    if backend=='python' and itk_threads!=1:
        raise RuntimeError('Actual SimpleITK inner thread count differs from1.')


def sitk_images(state,values,mask,affine):
    np,sitk=state['np'],state['sitk']
    image=sitk.GetImageFromArray(np.asarray(values,dtype=np.float64));roi=sitk.GetImageFromArray(np.asarray(mask,dtype=np.uint8))
    ndim=values.ndim;affine=np.asarray(affine);spacing=np.linalg.norm(affine[:3,:3],axis=0)
    flip=np.diag([-1.,-1.,1.]);direction=flip@affine[:3,:3]/spacing
    for obj in (image,roi):
        obj.SetSpacing(spacing[:ndim].tolist());obj.SetOrigin((flip@affine[:3,3])[:ndim].tolist())
        obj.SetDirection(direction[:ndim,:ndim].ravel().tolist())
    return image,roi


def prepared_call(state,output):
    np,pd=state['np'],state['pd'];request,plan=state['request'],state['plan']
    dataset=state['dataset'];family=request['case'].get('family')
    values,mask,affine=state['values'],state['mask'],dataset['affine_ras_mm']
    rows=[]
    for i in range(request['case']['roi_count']):
        identifier=f"{dataset['id']}-R{i:04}"
        if request['backend']=='rust':
            rad=state['rad'];image=rad.image.from_array(values,affine=affine,spatial_unit='mm')
            roi=rad.mask.from_array(mask,geometry=image.geometry)
            sample=rad.ImageSample(dict(sample_id=identifier,patient_id=f'P{i:04}'),image,roi)
            table=sample.extract(state['native_extraction'],**({} if family is None else {'families':[family]}))
            rows.append(table.features)
        else:
            row={'ID':identifier}
            if family!='shape_basic':
                image,roi=sitk_images(state,values,mask,affine)
                for name,klass in state['reference_classes'].items():
                    if family is not None and name!=family:
                        continue
                    engine=klass(image,roi,**state['reference_settings'])
                    for feature in plan['reference']['features'][name]:
                        engine.enableFeatureByName(feature)
                    row.update({'original_'+name+'_'+key:float(value) for key,value in engine.execute().items()})
            if family in (None,'shape_basic'):
                row.update(shape_basic_spec(mask,affine))
            rows.append(pd.DataFrame([row]))
    frame=pd.concat(rows,ignore_index=True)
    columns=expected_columns(plan,values.ndim,family)
    if set(frame.columns)!=set(['ID']+columns):
        raise RuntimeError('Prepared public API emitted an extra/missing feature; never filter an unmatched larger family silently.')
    return dict(features=frame[['ID']+columns].sort_values('ID').reset_index(drop=True),diagnostics=None,
        cache_summary=None,implementation_sha256=state.get('implementation_sha256'),output_directory=None)


def reference_preparation(state,image,roi,affine):
    np,sitk=state['np'],state['sitk'];affine=np.asarray(affine,dtype=float).copy()
    for step in state['request']['case']['preparation']:
        if step['operation']=='clip_range':
            values=np.clip(sitk.GetArrayFromImage(image).astype(float),step['lower'],step['upper'])
            clipped=sitk.GetImageFromArray(values);clipped.CopyInformation(image);image=clipped
        elif step['operation']=='crop_voxels':
            coords=np.argwhere(sitk.GetArrayViewFromImage(roi)==1)
            ndim=image.GetDimension();margin=np.asarray(step['margin_voxels'][:ndim])
            minimum=np.maximum(coords.min(axis=0)[::-1]-margin,0)
            maximum=np.minimum(coords.max(axis=0)[::-1]+margin,np.asarray(image.GetSize())-1)
            size=maximum-minimum+1;index=minimum.astype(int).tolist()
            image=sitk.RegionOfInterest(image,size.astype(int).tolist(),index)
            roi=sitk.RegionOfInterest(roi,size.astype(int).tolist(),index)
            offset=np.zeros(3);offset[:ndim]=minimum;affine[:3,3]+=affine[:3,:3]@offset
        else:
            raise RuntimeError('Unreviewed reference preprocessing operation.')
    return image,roi,affine


def e2e_call(state,output):
    pd=state['pd'];request,plan=state['request'],state['plan'];case=request['case']
    directory=Path(request['fixtures'])/state['dataset']['directory'];cohort=directory/f"cohort-{case['roi_count']}.csv"
    if request['backend']=='rust':
        rad=state['rad'];samples=rad.read_cohort(cohort)
        config=dict(extraction=state['native_extraction'],labels=[1],preparation=case['preparation'])
        result=rad.extract(samples,params=config,backend='rust',n_jobs=case['threads'],cache=request.get('cache_directory'))
        if result.config['failures'] or len(result.features)!=case['roi_count']:
            raise RuntimeError('Native batch did not retain every declared ROI as a successful feature row.')
        expected=request['native_build']['core_implementation_sha256']
        if result.config['implementation_sha256']!=expected:
            raise RuntimeError('Runtime implementation digest differs from the fixed native build manifest.')
        result.save(output)
        return dict(features=result.features,diagnostics=result.diagnostics,
            cache_summary=result.config.get('cache_summary'),implementation_sha256=result.config['implementation_sha256'],output_directory=str(output))
    from concurrent.futures import ThreadPoolExecutor
    samples=pd.read_csv(cohort,dtype={'sample_id':str,'patient_id':str})
    def one(row):
        image=state['sitk'].ReadImage(str(directory/row['image']));roi=state['sitk'].ReadImage(str(directory/row['mask']))
        image,roi,affine=reference_preparation(state,image,roi,state['dataset'].get('source_affine_ras_mm',state['dataset']['affine_ras_mm']))
        values=state['extractor'].execute(image,roi,label=1)
        features={'ID':row['sample_id']};diagnostic={'ID':row['sample_id'],'patient_id':row['patient_id'],'ok':True,'error':''}
        for name,value in values.items():
            if name.startswith('diagnostics_'):
                diagnostic[name]=str(value)
            else:
                features[name]=float(value)
        features.update(shape_basic_spec(state['sitk'].GetArrayViewFromImage(roi),affine))
        return features,diagnostic
    with ThreadPoolExecutor(max_workers=case['threads']) as executor:
        results=list(executor.map(one,samples.to_dict('records')))
    columns=expected_columns(plan,len(state['dataset']['shape_xyz']))
    frame=pd.DataFrame([r[0] for r in results])
    if set(frame.columns)!=set(['ID']+columns):
        raise RuntimeError('Reference emitted an extra/missing feature outside the explicit matched scope.')
    frame=frame[['ID']+columns].sort_values('ID').reset_index(drop=True)
    diagnostics=pd.DataFrame([r[1] for r in results]).sort_values('ID').reset_index(drop=True)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    frame.to_csv(output/'features.csv',index=False);diagnostics.to_csv(output/'diagnostics.csv',index=False)
    write_json(output/'config.json',dict(reference_settings=state['extractor'].settings,enabled_features=state['extractor'].enabledFeatures,
        preparation=case['preparation'],pyradiomics_version=state['environment']['pyradiomics_version'],samples=samples.to_dict('records'),shape_basic='independent project voxel-support specification'))
    return dict(features=frame,diagnostics=diagnostics,cache_summary=None,implementation_sha256=None,output_directory=str(output))


def artifact_from_frame(frame,columns,count):
    if list(frame.columns)!=['ID']+columns or len(frame)!=count or frame.ID.isna().any() or frame.ID.duplicated().any():
        raise RuntimeError('Exact declared feature order or sample identity/row count failed.')
    return dict(schema='radiomics.benchmark.v2.values.v1',sample_ids=frame.ID.astype(str).tolist(),
        feature_names=columns,values=clean(frame[columns].to_numpy(dtype=float)))


def worker(request):
    # Hash/source/input audits happen in coordinator preflight outside cold wall
    # time. The measured worker receives the already approved control snapshot.
    plan=request['plan_contents']
    state=worker_setup(request,plan)
    try:
        return worker_execute(request,plan,state)
    finally:
        state['thread_limiter'].restore_original_limits()


def worker_execute(request,plan,state):
    if request.get('describe_only'):
        return dict(status='ok',environment=state['environment'])
    dataset=request['dataset_contents']
    directory=Path(request['fixtures'])/dataset['directory']
    state['dataset']=dataset
    if request['case']['level'] in ('L1','L2'):
        state.update(values=state['np'].load(directory/'values.npy',allow_pickle=False),mask=state['np'].load(directory/'mask.npy',allow_pickle=False))
    call=prepared_call if request['case']['level'] in ('L1','L2') else e2e_call
    observation=Path(request['observation_directory']);observation.mkdir(parents=True,exist_ok=False)
    if request.get('validation_only'):
        # Correctness preflight is explicitly untimed; it is not another sample
        # in the seven-pair performance series and cannot produce a speed claim.
        result=call(state,observation/'validation')
        return dict(status='ok',validation_only=True,comparison_values=artifact_from_frame(result['features'],
            expected_columns(plan,len(dataset['shape_xyz']),request['case'].get('family')),request['case']['roi_count']),
            peak_memory_bytes=peak_resident_bytes(),environment_snapshot_sha256=request['environment_snapshot_sha256'],
            observed_thread_libraries=state['observed_thread_libraries'],cache_summary=result['cache_summary'],
            implementation_sha256=result['implementation_sha256'])
    warmup=[]
    for i in range(request['warmup_calls']):
        started=time.perf_counter();result=call(state,observation/f'warmup-{i}')
        duration=time.perf_counter()-started
        warm_artifact=artifact_from_frame(result['features'],expected_columns(plan,len(dataset['shape_xyz']),request['case'].get('family')),request['case']['roi_count'])
        warmup.append(dict(elapsed_seconds=duration,roi_count=request['case']['roi_count'],cache_summary=result['cache_summary'],
            comparison_values=warm_artifact))
        del result
    import gc
    gc.collect();started=time.perf_counter();result=call(state,observation/'timed')
    elapsed=time.perf_counter()-started
    artifact=artifact_from_frame(result['features'],expected_columns(plan,len(dataset['shape_xyz']),request['case'].get('family')),request['case']['roi_count'])
    cache=result['cache_summary'];mode=request['case']['state']
    if mode in ('cache_population','verified_cache_hit'):
        if cache is None or not cache['enabled']:
            raise RuntimeError('Native cache series requires actual core hit/miss instrumentation.')
        wanted_hits=request['case']['roi_count'] if mode=='verified_cache_hit' else 0
        if cache['hits']!=wanted_hits or cache['misses']!=request['case']['roi_count']-wanted_hits or cache['rejected'] or cache['write_failures']:
            raise RuntimeError('Measured cache hit/miss/rejection/write counts differ from the declared state.')
    peak=peak_resident_bytes()
    if peak is None or peak<=0:
        raise RuntimeError('The declared full-process resident-memory metric is unavailable.')
    return dict(status='ok' if peak<=plan['execution']['memory_budget_bytes'] else 'failed',
        error=None if peak<=plan['execution']['memory_budget_bytes'] else 'OperationalFailure: process peak RSS exceeded the fixed 2GiB memory budget.',
        elapsed_seconds=elapsed,peak_memory_bytes=peak,comparison_values=artifact,
        warmup=warmup,timed_roi_count=request['case']['roi_count'],warmup_roi_count=request['warmup_calls']*request['case']['roi_count'],
        cache_summary=cache,implementation_sha256=result['implementation_sha256'],environment_snapshot_sha256=request['environment_snapshot_sha256'],
        effective_native_extraction=state['native_extraction'],effective_reference_settings=state['reference_settings'],
        observed_thread_libraries=state['observed_thread_libraries'],simpleitk_effective_threads=state['environment'].get('simpleitk_effective_threads'),
        dimension_binding_sha256=request.get('dimension_binding_sha256'),
        stage_seconds=None,stage_status='Full declared public call measured; no invented per-stage instrumentation.',
        allocation_count=None,allocation_bytes=None,allocation_status='Unmeasured; peak RSS is a process metric, not an allocation count.')


def wheel_package_files(path):
    with zipfile.ZipFile(path) as archive:
        return {name[len('ov_radiomics/'):]:hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
            if name.startswith('ov_radiomics/') and Path(name).suffix.lower() in ('.py','.pyd','.so','.dll')}


def physical_core_count():
    if sys.platform=='win32':
        import ctypes
        size=ctypes.c_ulong(0);function=ctypes.windll.kernel32.GetLogicalProcessorInformationEx
        function(0,None,ctypes.byref(size))
        if not size.value:
            return None
        buffer=ctypes.create_string_buffer(size.value)
        if not function(0,buffer,ctypes.byref(size)):
            return None
        position=count=0
        while position<size.value:
            relation,length=struct.unpack_from('<II',buffer.raw,position)
            if length<8 or position+length>size.value:
                return None
            count+=relation==0;position+=length
        return count
    try:
        groups=set()
        for processor in Path('/proc/cpuinfo').read_text(encoding='utf-8').split('\n\n'):
            values={line.split(':',1)[0].strip():line.split(':',1)[1].strip() for line in processor.splitlines() if ':' in line}
            if 'physical id' in values and 'core id' in values:
                groups.add((values['physical id'],values['core id']))
        return len(groups) or None
    except OSError:
        return None


def machine_details(fixtures,output):
    source=machine_manifest(ROOT)
    details={key:source[key] for key in ('platform','cpu','logical_cpus','total_physical_memory_bytes','file_cache_control','power_mode','background_load','missing_provenance')}
    details['physical_cpus']=physical_core_count()
    details['storage']=dict(input_root=str(fixtures),output_root=str(output),
        output_volume_capacity_bytes=shutil.disk_usage(output).total,output_volume_free_bytes=shutil.disk_usage(output).free,
        device_model='unrecorded; no I/O-device-speed claim')
    if details['physical_cpus'] is not None:
        details['missing_provenance']=[key for key in details['missing_provenance'] if key!='physical core count']
    if sys.platform=='win32':
        try:
            raw_mode=subprocess.check_output(['powercfg','/getactivescheme'],timeout=10,stderr=subprocess.DEVNULL)
            mode,encoding=decode_windows_metadata(raw_mode)
            details['power_mode']=mode;details['missing_provenance']=[key for key in details['missing_provenance'] if key!='power mode']
            details['power_mode_output_encoding']=encoding;details['power_mode_raw_hex']=raw_mode.hex()
        except (OSError,subprocess.SubprocessError,UnicodeError,TypeError):
            pass
    details['background_load']='Other host processes not controlled by runner; all seven fixed observations retained and spread reported.'
    return details


def decode_windows_metadata(raw):
    if not isinstance(raw,bytes):
        raise TypeError('Windows metadata command must be captured as bytes.')
    encodings=['utf-16'] if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else ['utf-16-le','utf-8','gb18030'] if b'\0' in raw[:64] else ['utf-8','gb18030']
    for encoding in encodings:
        try:
            return raw.decode(encoding).strip(),encoding
        except UnicodeError:
            continue
    return raw.decode('utf-8',errors='replace').strip(),'utf-8-replacement; original bytes retained'


def cases(plan,manifest,ct_plan=None):
    output=[];datasets={d['id']:d for d in manifest['datasets']}
    for declaration in plan['datasets']:
        dataset=datasets[declaration['id']]
        for level in declaration['levels']:
            config=plan['levels'][level]
            for family in config['families'] if level=='L1' else [None]:
                for count in config.get('roi_counts',[config.get('roi_count_per_call',1)]):
                    for threads in config['threads']:
                        for state in config.get('states',[config.get('state')]):
                            prep=[]
                            if level=='L3':
                                for step in config['preparation']:
                                    if step['operation']=='clip_range':
                                        prep.append(dict(operation='clip_range',lower=step['lower'],upper=float(declaration.get('gray_levels',6))))
                                    else:
                                        prep.append(step)
                            operation=('prepared_'+family if family is not None else 'prepared_all55_shape_basic' if level=='L2' else 'e2e_all55_shape_basic')
                            case=dict(suite='radiomics',level=level,dataset=dataset['id'],family=family,operation=operation,state=state,
                                threads=threads,roi_count=count,preparation=prep,input_sha256=dataset['input_sha256'],
                                source_roi_voxels=dataset['roi_voxels'],source_gray_values=dataset['present_gray_values'],
                                output_feature_names=expected_columns(plan,len(dataset['shape_xyz']),family))
                            case['configuration_sha256']=json_hash(case)
                            case['id']=f"{level}-{dataset['id']}-{family or 'full'}-n{count}-j{threads}-{state}"
                            output.append(case)
    if ct_plan is not None:
        dataset=datasets[ct_plan['dataset']['id']]
        for declaration in ct_plan['cases']:
            case=dict(suite='radiomics',level=declaration['level'],dataset=dataset['id'],family=None,
                operation='prepared_all55_shape_basic' if declaration['level']=='L2' else 'e2e_all55_shape_basic',state=declaration['state'],
                threads=declaration['threads'],roi_count=declaration['roi_count'],preparation=declaration['preparation'],input_sha256=dataset['input_sha256'],
                source_roi_voxels=dataset['roi_voxels'],source_gray_values=dataset['present_gray_values'],
                native_extraction=ct_plan['native_extraction'],output_feature_names=expected_columns(plan,3),
                frozen_reference='bbox_crop_reference.json',ct_supplement_sha256=manifest['ct_supplement_sha256'])
            case['configuration_sha256']=json_hash(case);case['id']=declaration['id'];output.append(case)
        if len(output)!=ct_plan['total_primary_cases'] or len({case['id'] for case in output})!=len(output):
            raise ValueError('CT append changed/dropped/duplicated the unchanged base grid.')
    return output


def frozen_reference_provenance(plan):
    directory=ROOT/'golden/python_golden_v1';manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    path=directory/'reference.json'
    if sha256_file(path)!=manifest['files_sha256']['reference.json']:
        raise RuntimeError('Frozen reference provenance bytes differ from the accepted Golden manifest.')
    environment=json.loads(path.read_text(encoding='utf-8'))['environment']
    if environment['pyradiomics_revision']!=plan['reference']['pyradiomics_revision']:
        raise RuntimeError('Frozen public reference revision differs from the predeclared benchmark.')
    return dict(revision=environment['pyradiomics_revision'],files=environment['pyradiomics_files_sha256'],
        golden_reference_sha256=sha256_file(path))


def audit_observation_inputs(request):
    """All expensive byte audits are outside the owned worker's cold wall clock."""
    load_plan(request['plan'],request['approved_plan_sha256'])
    if request.get('describe_only'):
        return dict(probe=True)
    environment=request['audit_environment'];directory=Path(environment['package_path'])
    actual={str(path.relative_to(directory)).replace('\\','/'):sha256_file(path) for path in sorted(directory.rglob('*'))
        if path.is_file() and path.suffix.lower() in ('.py','.pyd','.so','.dll')}
    if actual!=environment['package_files_sha256']:
        raise RuntimeError('Source/installed package bytes changed outside the immutable preflight snapshot.')
    for path,digest in environment['dependency_binary_files_sha256'].items():
        if sha256_file(path)!=digest:
            raise RuntimeError('A compiled numerical dependency changed after non-timed preflight.')
    dataset=request['dataset_contents'];root=Path(request['fixtures'])/dataset['directory']
    for name,digest in dataset['files_sha256'].items():
        if sha256_file(root/name)!=digest:
            raise RuntimeError('Prepared input bytes changed after non-timed preflight.')
    return dict(package_sha256=json_hash(actual),input_sha256=dataset['input_sha256'],files_verified=len(actual)+len(dataset['files_sha256']),
        dependency_binary_files_sha256=environment['dependency_binary_files_sha256'])


def merge_peak_memory(worker_peak,coordinator_peak):
    values=[value for value in (worker_peak,coordinator_peak) if isinstance(value,(int,float)) and value>0]
    return int(max(values)) if values else None


def finish_worker_transport(result,request,footer_peak):
    """Numerical artifact persistence/digests are parent postflight, not cold API work."""
    if request.get('describe_only'):
        return result
    directory=Path(request['observation_directory'])
    artifact=result.pop('comparison_values',None)
    if artifact is not None:
        directory.mkdir(parents=True,exist_ok=True);write_json(directory/'comparison.json',artifact)
        result.update(comparison_artifact=str(directory/'comparison.json'),output_digest=json_hash(artifact))
    for i,item in enumerate(result.get('warmup',[])):
        values=item.pop('comparison_values',None)
        if values is not None:
            write_json(directory/f'warmup-{i}-comparison.json',values)
            item.update(comparison_artifact=str(directory/f'warmup-{i}-comparison.json'),output_digest=json_hash(values))
            if values!=artifact:
                result.update(status='failed',error='Warmup and timed calls differ within the fixed worker; reproducibility failed.')
    files={str(path.relative_to(directory)).replace('\\','/'):dict(bytes=path.stat().st_size,sha256=sha256_file(path))
        for path in sorted(directory.rglob('*')) if path.is_file()} if directory.exists() else {}
    result.update(output_files=files,output_bytes=sum(item['bytes'] for name,item in files.items() if name.startswith('timed/')))
    return result


def launch_worker(executable,request,request_path,stdout_path,stderr_path,budget,timeout):
    """Only this runner's subprocess is monitored/killed; raw errors are retained."""
    try:
        preflight=audit_observation_inputs(request)
    except Exception as exc:
        return dict(status='failed',error=f'PreflightAuditFailure: {type(exc).__name__}: {exc}',
            peak_memory_bytes=None,process_wall_seconds=None,worker_started=False)
    payload={key:value for key,value in request.items() if key!='audit_environment'}
    write_json(request_path,payload)
    environment=dict(os.environ)
    environment.pop('PYTHONPATH',None);environment.pop('PYTHONHOME',None)
    plan,_=load_plan(request['plan'],request['approved_plan_sha256'])
    environment.update(plan['execution']['thread_environment']);environment['MPLBACKEND']='Agg'
    environment['PYTHONDONTWRITEBYTECODE']='1'
    observed_peak=0;sample_count=0;enforced=None;process=None;started=None
    try:
        with Path(stdout_path).open('xb') as out,Path(stderr_path).open('xb') as err:
            started=time.perf_counter()
            process=subprocess.Popen([str(executable),str(Path(__file__).resolve()),'--worker',str(request_path)],
                cwd=ROOT,env=environment,stdout=out,stderr=err)
            while process.poll() is None:
                resident,peak=process_memory(process.pid)
                if resident is not None:
                    observed_peak=max(observed_peak,resident,peak or 0);sample_count+=1
                    if resident>budget or (peak is not None and peak>budget):
                        enforced='OperationalFailure: owned worker exceeded the fixed process-resident memory budget.'
                        process.kill();break
                if time.perf_counter()-started>timeout:
                    enforced='OperationalFailure: owned worker exceeded the predeclared observation timeout.'
                    process.kill();break
                time.sleep(.02)
            process.wait();wall=time.perf_counter()-started
    except Exception as exc:
        cleanup_error=None
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
                process.wait()
            except Exception as cleanup:
                cleanup_error=f'{type(cleanup).__name__}: {cleanup}'
        return dict(status='failed',error=f'OwnedWorkerLaunchOrMonitorFailure: {type(exc).__name__}: {exc}',
            cleanup_error=cleanup_error,worker_started=process is not None,worker_exit_code=None if process is None else process.returncode,
            process_wall_seconds=None if started is None else time.perf_counter()-started,
            peak_memory_bytes=observed_peak or None,stdout_path=str(stdout_path),stderr_path=str(stderr_path))
    # Wall clock is stopped before file audits, protocol parsing, comparisons,
    # canonical sidecar persistence and output-file re-reading/digests.
    try:
        messages=[json.loads(line) for line in Path(stdout_path).read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(messages)!=2 or any(not isinstance(message,dict) for message in messages):
            raise ValueError('Expected one numerical result and one lifetime-memory footer.')
        result=messages[0];footer_peak=messages[1].get('worker_peak_after_transport_bytes')
        if type(footer_peak) is not int or footer_peak<=0:
            raise ValueError('Lifetime-memory footer is missing or invalid; do not underreport final transport stage.')
    except Exception as exc:
        result=dict(status='failed',error=f'Worker protocol failure: {type(exc).__name__}: {exc}');footer_peak=None
    worker_peak=merge_peak_memory(result.get('peak_memory_bytes'),footer_peak)
    result['worker_peak_memory_bytes']=worker_peak
    result['peak_memory_bytes']=merge_peak_memory(worker_peak,observed_peak)
    result.update(process_wall_seconds=wall,worker_exit_code=process.returncode,
        coordinator_peak_resident_bytes=observed_peak or None,resident_poll_samples=sample_count,
        resident_poll_seconds=.02,stdout_path=str(stdout_path),stderr_path=str(stderr_path))
    if process.returncode!=0 or enforced is not None:
        result['status']='failed';result['error']=enforced or result.get('error') or f'Worker exit {process.returncode}'
    if result.get('peak_memory_bytes') is not None and result['peak_memory_bytes']>budget:
        result.update(status='failed',error='OperationalFailure: full observed lifetime peak exceeded fixed memory budget.')
    try:
        postflight=audit_observation_inputs(request)
        result['non_timed_audits']=dict(before=preflight,after=postflight)
        result=finish_worker_transport(result,request,footer_peak)
    except Exception as exc:
        result.update(status='failed',error=f'NonTimedAuditOrTransportFailure: {type(exc).__name__}: {exc}')
    if request.get('case',{}).get('state')=='cold_process_total':
        result['api_elapsed_seconds']=result.get('elapsed_seconds');result['elapsed_seconds']=wall
    return result


def validate_build_manifest(build,wheel,base):
    def hex_digest(value,length):
        return isinstance(value,str) and len(value)==length and all(c in '0123456789abcdef' for c in value)
    if build.get('schema')!='radiomics.native-build.v1' or build.get('source_tree_clean') is not True:
        raise ValueError('A clean, versioned native build receipt is mandatory; empty/unverified manifests are not accepted.')
    for key,length in [('source_revision',40),('source_tree',40),('cargo_lock_sha256',64),('core_implementation_sha256',64)]:
        if not hex_digest(build.get(key),length):
            raise ValueError('Missing/invalid required native build provenance: '+key)
    for key in ('rustc_verbose','rust_target','python_version','maturin_version'):
        if not isinstance(build.get(key),str) or not build[key].strip():
            raise ValueError('Missing required native build provenance: '+key)
    if build.get('profile')!=dict(name='release',opt_level=3,lto='thin',codegen_units=1):
        raise ValueError('The frozen native receipt must identify the declared optimized release profile.')
    flags=build.get('rust_flags',{})
    if any(key not in flags or flags[key] is not None and not isinstance(flags[key],str) for key in ('RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS')):
        raise ValueError('Both explicit native Rust flag states are required.')
    commands=build.get('build_commands')
    if not isinstance(commands,list) or not commands or any(not isinstance(cmd,list) or not cmd or any(not isinstance(arg,str) or not arg for arg in cmd) for cmd in commands):
        raise ValueError('Native build commands must be retained as nonempty argv lists.')
    for key in ('wheel','cli'):
        artifact=build.get(key,{})
        if not isinstance(artifact.get('path'),str) or not hex_digest(artifact.get('sha256'),64):
            raise ValueError('Native receipt requires path and SHA256 for '+key)
        path=Path(artifact['path']);path=path if path.is_absolute() else Path(base)/path
        if not path.is_file() or sha256_file(path)!=artifact['sha256']:
            raise ValueError('Native '+key+' artifact differs from its mandatory receipt.')
    if sha256_file(wheel)!=build['wheel']['sha256']:
        raise ValueError('Requested wheel bytes differ from the receipt.')


def package_snapshot(args,output):
    wheel=Path(args.native_wheel).resolve();receipt=Path(args.native_build_manifest).resolve();build=json.loads(receipt.read_text(encoding='utf-8'))
    validate_build_manifest(build,wheel,receipt.parent)
    recorded_tree=subprocess.check_output(['git','rev-parse',build['source_revision']+'^{tree}'],cwd=ROOT,text=True).strip()
    if recorded_tree!=build['source_tree']:
        raise ValueError('Native build source revision/tree differ from the repository object provenance.')
    directory=output/'snapshot';directory.mkdir()
    copy=directory/wheel.name;shutil.copyfile(wheel,copy)
    write_json(directory/'native-build.json',build)
    return dict(native_build=build,native_wheel=dict(path=str(copy),sha256=sha256_file(copy),package_files_sha256=wheel_package_files(copy)),
        benchmark_runner=dict(path=str(Path(__file__).resolve()),sha256=sha256_file(__file__),
            protocol_sha256=sha256_file(Path(__file__).with_name('benchmark_protocol.py'))))


def observation_base(args,snapshot,case,backend,output):
    environment=snapshot['audited_environments'][backend]
    return dict(plan=str(Path(args.plan).resolve()),approved_plan_sha256=args.approved_plan_sha256,
        fixtures=str(Path(args.fixtures).resolve()),case=case,backend=backend,
        plan_contents=snapshot['plan_contents'],dataset_contents=snapshot['dataset_contents'][case['dataset']],
        native_build={'core_implementation_sha256':snapshot['native_build']['core_implementation_sha256']},
        audit_environment=environment,expected_package_path=environment['package_path'],
        preflight_environment={'versions':environment['versions']},environment_snapshot_sha256=json_hash(environment),
        dimension_binding_sha256=snapshot.get('dimension_binding_sha256'),
        observation_directory=str(output),warmup_calls=0 if case['state'] in ('cold_process_total','cache_population') else 1)


def raw_observation(case,backend,pair,order,result):
    result=dict(result)
    result.update({key:case[key] for key in ('suite','level','dataset','operation','state','threads','input_sha256','configuration_sha256')})
    result.update(case_id=case['id'],family=case.get('family'),pair=pair,order=order,roi_count=case['roi_count'],
        correctness_checked=False,roi_per_second=None if not result.get('elapsed_seconds') else case['roi_count']/result['elapsed_seconds'],backend=backend)
    return result


def summarize_case(case,records,plan):
    summary=paired_summary(records,repeats=plan['execution']['paired_repetitions'])
    summary.update(case=case)
    if not summary['valid_speed_comparison']:
        return summary
    memory={backend:latency_summary([float(r['peak_memory_bytes']) for r in records if r['backend']==backend]) for backend in ('python','rust')}
    # latency_summary supplies identical declared quantiles; rename units explicitly.
    memory={backend:{key.replace('_seconds','_bytes'):value for key,value in values.items() if key!='p95_interpretation'} for backend,values in memory.items()}
    ratio=memory['rust']['median_bytes']/memory['python']['median_bytes']
    summary.update(peak_memory=memory,peak_memory_comparison=ratio,memory_status='measured full-process lifetime peak resident memory',
        memory_engineering_threshold=plan['performance_reporting']['memory_engineering_ratio_maximum'],memory_engineering_miss=ratio>plan['performance_reporting']['memory_engineering_ratio_maximum'],
        roi_per_second={backend:case['roi_count']/summary['backends'][backend]['median_seconds'] for backend in ('python','rust')})
    if case['level']=='L2':
        runtime=1/summary['ratio_python_over_rust']
        summary.update(native_over_reference_runtime_ratio=runtime,L2_engineering_runtime_miss=runtime>1.10,
            speedup_target=1.5,speedup_target_met=summary['ratio_python_over_rust']>=1.5)
    elif case['level']=='L3' and case['state']=='warm_uncached':
        summary.update(speedup_target=1.5,speedup_target_met=summary['ratio_python_over_rust']>=1.5)
    return summary


def append_record(path,record):
    with Path(path).open('a',encoding='utf-8',newline='\n') as handle:
        handle.write(json.dumps(clean(record),allow_nan=False,sort_keys=True)+'\n');handle.flush();os.fsync(handle.fileno())


def frozen_ct_artifact(plan,manifest,fixtures,output):
    matches=[dataset for dataset in manifest['datasets'] if dataset.get('ct_supplement_sha256')]
    if not matches:
        return {}
    dataset=matches[0];path=Path(fixtures)/dataset['directory']/'bbox_crop_reference.json'
    if sha256_file(path)!=dataset['reference_bindings']['tight_bbox_reference']['sha256']:
        raise ValueError('Frozen CT crop reference changed before measurement.')
    reference=json.loads(path.read_text(encoding='utf-8'));values={**reference['features'],**reference['basic_shape']}
    columns=expected_columns(plan,3)
    if set(values)!=set(columns):
        raise ValueError('Frozen CT scalar/basic-shape reference does not cover the exact66-column output.')
    artifact=dict(schema='radiomics.benchmark.v2.values.v1',sample_ids=[dataset['id']+'-R0000'],feature_names=columns,values=[[values[name] for name in columns]])
    destination=Path(output)/'frozen-ct-comparison.json';write_json(destination,artifact)
    return {dataset['id']:str(destination)}


def run(args,plan,plan_sha):
    output=Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Measurement output must be fresh; no repeated favorable-run overwrite.')
    fixtures=Path(args.fixtures).resolve();manifest=json.loads((fixtures/'manifest.json').read_text(encoding='utf-8'))
    if manifest['approved_plan_sha256']!=plan_sha:
        raise ValueError('Input corpus was prepared from different plan bytes.')
    ct_plan=ct_sha=None
    if args.ct_plan:
        ct_plan,ct_sha=load_ct_plan(args.ct_plan,plan_sha,args.approved_ct_plan_sha256)
        if manifest.get('ct_supplement_sha256')!=ct_sha:
            raise ValueError('Prepared public CT supplement differs from its pre-measurement approval.')
    elif manifest.get('ct_supplement_sha256'):
        raise ValueError('A CT corpus requires its exact approved supplemental plan, not an implicit extra dataset.')
    binding,binding_sha=load_dimension_binding(args.dimension_binding,plan_sha,ct_sha,args.approved_dimension_binding_sha256)
    output.mkdir(parents=True);snapshot=package_snapshot(args,output)
    write_json(output/'plan.json',plan);write_json(output/'input-manifest.json',manifest)
    if ct_plan is not None:
        write_json(output/'ct-plan.json',ct_plan)
    write_json(output/'dimension-binding.json',binding)
    environments={}
    reference_expected=frozen_reference_provenance(plan)
    for backend,executable in [('python',args.reference_python),('rust',args.native_python)]:
        request=dict(plan=str(Path(args.plan).resolve()),plan_contents=plan,approved_plan_sha256=plan_sha,backend=backend,describe_only=True,
            expected_reference=reference_expected,
            expected_native_package_files=snapshot['native_wheel']['package_files_sha256'],native_build=snapshot['native_build'])
        result=launch_worker(executable,request,output/f'probe-{backend}.json',output/f'probe-{backend}.stdout.json',output/f'probe-{backend}.stderr.txt',
            plan['execution']['memory_budget_bytes'],plan['execution']['maximum_observation_seconds'])
        environments[backend]=result
        if result['status']!='ok':
            write_json(output/'preflight-failed.json',dict(status='blocked_before_timing',backend=backend,probe=result))
            raise RuntimeError('Backend preflight failed before measurement; inspect retained probe output.')
    # Coordinator package state is not the measured native wheel/source identity.
    machine=machine_details(fixtures,output)
    write_json(output/'provenance.json',dict(machine=machine,environments=environments,native_snapshot=snapshot,
        plan_sha256=plan_sha,ct_plan_sha256=ct_sha,dimension_binding_sha256=binding_sha,input_manifest_sha256=sha256_file(fixtures/'manifest.json'),
        benchmark_runner_source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        command_arguments=vars(args)))
    snapshot['audited_environments']={backend:result['environment'] for backend,result in environments.items()}
    snapshot['plan_contents']=plan;snapshot['dataset_contents']={dataset['id']:dataset for dataset in manifest['datasets']}
    snapshot['dimension_binding_sha256']=binding_sha
    summaries=[];all_records=[];baseline={};ct_expected=frozen_ct_artifact(plan,manifest,fixtures,output)
    declared=bind_inactive_axis(cases(plan,manifest,ct_plan),binding)
    for case_index,case in enumerate(declared):
        case_directory=output/case['id'];case_directory.mkdir();records=[]
        for pair in range(plan['execution']['paired_repetitions']):
            pair_records={}
            for order,backend in enumerate(('python','rust') if pair%2==0 else ('rust','python')):
                run_directory=case_directory/f'pair-{pair:02}-{backend}'
                request=observation_base(args,snapshot,case,backend,run_directory)
                result=launch_worker(args.reference_python if backend=='python' else args.native_python,request,
                    case_directory/f'pair-{pair:02}-{backend}.request.json',case_directory/f'pair-{pair:02}-{backend}.stdout.json',case_directory/f'pair-{pair:02}-{backend}.stderr.txt',
                    plan['execution']['memory_budget_bytes'],plan['execution']['maximum_observation_seconds'])
                result['dimension_binding_sha256']=binding_sha
                pair_records[backend]=raw_observation(case,backend,pair,order,result)
            comparison=dict(passed=False,reason='At least one backend execution failed; retain timings as diagnostic only.')
            if all(r['status']=='ok' for r in pair_records.values()):
                comparison=safe_compare_artifacts(pair_records['python'].get('comparison_artifact'),pair_records['rust'].get('comparison_artifact'),plan)
                if case['dataset'] in ct_expected:
                    frozen={backend:safe_compare_artifacts(ct_expected[case['dataset']],result.get('comparison_artifact'),plan) for backend,result in pair_records.items()}
                    comparison=dict(passed=comparison['passed'] and all(row['passed'] for row in frozen.values()),
                        paired_compatibility=comparison,against_frozen_pre_native_crop_reference=frozen)
            for record in pair_records.values():
                record['correctness_checked']=bool(comparison['passed']);record['comparison']=comparison
                records.append(record);all_records.append(record);append_record(output/'raw-observations.jsonl',record)
            if comparison['passed'] and case['level']=='L3' and case['state']=='warm_uncached' and pair==0:
                baseline[case['dataset']]=pair_records['rust']['comparison_artifact']
        summary=summarize_case(case,records,plan);summaries.append(summary)
        write_json(case_directory/'summary.json',summary)
        append_record(output/'progress.jsonl',dict(completed_case=case['id'],case_index=case_index+1,total_cases=len(declared),
            valid_speed_comparison=summary['valid_speed_comparison'],ratio_python_over_rust=summary.get('ratio_python_over_rust')))
        print(f"[{case_index+1}/{len(declared)}] {case['id']}: "+str(summary.get('ratio_python_over_rust','invalid')),flush=True)
    cache_records=run_cache_series(args,plan,snapshot,manifest,output,baseline)
    statuses=overall_status(summaries,cache_records,all_records,expected_case_count=len(declared),repeats=plan['execution']['paired_repetitions'])
    report=dict(schema='radiomics.benchmark.v2.report.v1',suite='radiomics',plan_sha256=plan_sha,
        ct_plan_sha256=ct_sha,dimension_binding_sha256=binding_sha,expected_primary_cases=len(declared),expected_primary_backend_observations=len(declared)*14,
        correctness_status=statuses['correctness_status'],operational_status=statuses['operational_status'],
        performance_policy='Targets/regression thresholds are disclosed separately from numerical correctness; no cross-suite total score.',
        performance_scope='Public API calls and described same-process batch workflows, not pure Rust kernels or a new blanket IBSI conformance claim.',
        machine=machine,native_build=snapshot['native_build'],results=summaries,cache_series=cache_records,
        raw_observations='raw-observations.jsonl',scaling=scaling_summaries(summaries),
        missing_evidence=['L5 CLI install/startup performance measured separately','allocator count/bytes unmeasured','per-stage extraction instrumentation unmeasured',
            'OS disk cache uncontrolled','power mode/background load not controlled by this runner'],
        p95_interpretation='Empirical linear quantiles, n=7 per backend/case; not a precise tail guarantee.')
    write_json(output/'report.json',report);write_report_markdown(output/'report.md',report)
    return report


def run_cache_series(args,plan,snapshot,manifest,output,baseline):
    declaration=plan['levels']['cache'];dataset=declaration['dataset']
    original=next(case for case in cases(plan,manifest) if case['level']=='L3' and case['dataset']==dataset and case['state']=='warm_uncached')
    directory=output/'native-cache-series';directory.mkdir();records=[]
    for pair in range(plan['execution']['paired_repetitions']):
        cache_dir=directory/f'cache-{pair:02}'
        for state in ('cache_population','verified_cache_hit'):
            case={**original,'state':state,'id':f'cache-{dataset}-{state}'}
            case['configuration_sha256']=json_hash({k:v for k,v in case.items() if k not in ('configuration_sha256','id')})
            request=observation_base(args,snapshot,case,'rust',directory/f'pair-{pair:02}-{state}')
            request['cache_directory']=str(cache_dir)
            result=launch_worker(args.native_python,request,directory/f'pair-{pair:02}-{state}.request.json',
                directory/f'pair-{pair:02}-{state}.stdout.json',directory/f'pair-{pair:02}-{state}.stderr.txt',
                plan['execution']['memory_budget_bytes'],plan['execution']['maximum_observation_seconds'])
            result['dimension_binding_sha256']=snapshot.get('dimension_binding_sha256')
            record=raw_observation(case,'rust',pair,0,result)
            comparison=dict(passed=False,reason='No valid uncached native baseline or cache-series execution failed.')
            if result['status']=='ok' and dataset in baseline:
                comparison=safe_compare_artifacts(baseline[dataset],result.get('comparison_artifact'),plan)
            record['correctness_checked']=comparison['passed'];record['comparison']=comparison
            records.append(record);append_record(output/'raw-observations.jsonl',record)
    summary=dict(backend='rust',comparative_speedup=False,population_and_hit_observations=records,
        policy='Seven independently populated caches then verified hit observations; population cost retained. Cached native output is never compared as a speedup against uncached Python.')
    for state in ('cache_population','verified_cache_hit'):
        chosen=[r for r in records if r['state']==state]
        summary[state]=dict(valid=all(r['status']=='ok' and r['correctness_checked'] for r in chosen),n=len(chosen))
        if summary[state]['valid']:
            summary[state].update(latency=latency_summary([r['elapsed_seconds'] for r in chosen]),
                peak_memory_bytes=[r['peak_memory_bytes'] for r in chosen],actual_cache_summaries=[r['cache_summary'] for r in chosen])
    write_json(directory/'summary.json',summary)
    return summary


def overall_status(summaries,cache,primary_records=None,*,expected_case_count=58,repeats=7):
    cache_rows=cache.get('population_and_hit_observations',[])
    complete=(len(cache_rows)==14 and len(summaries)==expected_case_count and primary_records is not None
        and len(primary_records)==expected_case_count*repeats*2)
    numerical=complete and all(summary.get('valid_speed_comparison') is True for summary in summaries) and all(
        row.get('status')=='ok' and row.get('correctness_checked') is True for row in cache_rows)
    records=(primary_records or [])+cache_rows
    operational=complete and primary_records is not None and len(primary_records)==expected_case_count*repeats*2 and all(row.get('status')=='ok' for row in records)
    return dict(correctness_status='passed_measured_applicable_cases' if numerical else 'failed_or_incomplete_measured_cases',
        operational_status='passed_recorded_observations' if operational else 'failed_or_incomplete_recorded_observations')


def scaling_summaries(summaries):
    output=[];grouped={}
    for summary in summaries:
        case=summary['case']
        if case['level']=='L4':
            grouped.setdefault((case['dataset'],case['roi_count']),{})[case['threads']]=summary
    for (dataset,count),group in sorted(grouped.items()):
        row=dict(dataset=dataset,roi_count=count,thread_grid=sorted(group),backends={})
        for backend in ('python','rust'):
            if set(group)=={1,4} and all(s['valid_speed_comparison'] for s in group.values()):
                single=group[1]['backends'][backend]['median_seconds'];parallel=group[4]['backends'][backend]['median_seconds']
                row['backends'][backend]=dict(single_thread_seconds=single,four_workers_seconds=parallel,
                    speedup_4_over_1=single/parallel,efficiency=(single/parallel)/4,
                    roi_per_second_single=count/single,roi_per_second_four=count/parallel,
                    stable_under_declared_2GiB_budget=True)
            else:
                row['backends'][backend]=dict(stable_under_declared_2GiB_budget=False,status='failed_or_incomplete_case; not dropped')
        output.append(row)
    return output


def write_report_markdown(path,report):
    lines=['# Radiomics Benchmark v2 measured report','',
        'This report covers only the predeclared Radiomics suite. Correctness, speed targets, memory regression checks and deployment evidence remain distinct.',
        '',f"Plan SHA256: `{report['plan_sha256']}`.",
        'L1 is a selected-family public array call, including conversion/binning and its actual matrix/scalar work. L2–L4 compute the exact 55-feature list plus the separately described project shape_basic descriptors on both backends. No mesh/filter feature is timed.',
        '', '| Case | Python median s | Rust median s | Python/Rust speedup | Rust/Python peak RSS | Correct matching output |',
        '|---|---:|---:|---:|---:|---|']
    for row in report['results']:
        if row['valid_speed_comparison']:
            lines.append(f"| {row['case']['id']} | {row['backends']['python']['median_seconds']:.6g} | {row['backends']['rust']['median_seconds']:.6g} | {row['ratio_python_over_rust']:.4g} | {row['peak_memory_comparison']:.4g} | passed applicable comparison |")
        else:
            lines.append(f"| {row['case']['id']} | diagnostic only | diagnostic only | withheld | inspect raw | failed/incomplete |")
    lines+=['','Each JSON summary retains n, q25/q75, IQR, empirical p95, min/max, paired ratios, ROI/s, process-resident peaks, source/toolchain provenance and all raw observations. Seven repetitions do not provide a precise tail estimate.',
        '', '## Missed engineering thresholds and targets','']
    misses=[r for r in report['results'] if r.get('L2_engineering_runtime_miss') or r.get('memory_engineering_miss') or r.get('speedup_target_met') is False]
    if misses:
        for row in misses:
            lines.append(f"- `{row['case']['id']}`: runtime regression={row.get('L2_engineering_runtime_miss',False)}, memory regression={row.get('memory_engineering_miss',False)}, >=1.5× speed target met={row.get('speedup_target_met','not applicable')}. Retain and investigate; do not change correctness tolerances or claim an algorithm failure merely because it is slower.")
    else:
        lines.append('No missed target among valid applicable comparisons; unmeasured/failed cases do not count as passes.')
    lines+=['','## Scope and limitations','',
        '- Raw records and all failed observations remain in `raw-observations.jsonl`; no cross-suite combined acceleration score is produced.',
        '- Batch sources intentionally repeat a disclosed synthetic image/ROI with unique identifiers. Content cache is disabled for uncached throughput. Native cache population/hits are a separate noncomparative series.',
        '- Peak RSS covers one complete worker process lifetime, including imports, fixed fixture buffers, warmup, timed calls and result-transport bookkeeping. Both backends use outer threads in that same process. These numbers are not Python traced allocations or per-call allocation counts.',
        '- Cold means fresh interpreter process, with startup/import/API/export/numerical-report transport retained. External package/input audits and artifact-file hashing occur outside the cold clock. It is not a flushed-disk-cache measurement or Python-free CLI startup result.',
        '- Observed scaling is limited to the frozen workers=1/4 and ROI-count grid under a 2GiB budget. It does not establish a theoretical maximum throughput.',
        '- Median/IQR/p95 and alternating raw pairs describe uncertainty. Gains within observed noise should not be advertised as acceleration.',
        '- This is not a new IBSI conformance claim, clinical validation or blanket project Release Gate pass. The independent IBSI matrix and Golden rules are unchanged.',
        '- '+ '; '.join(report['missing_evidence'])+'.', '']
    with Path(path).open('x',encoding='utf-8',newline='\n') as handle:
        handle.write('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default=str(ROOT/'benchmarks/v2/radiomics-plan.json'))
    parser.add_argument('--approved-plan-sha256')
    parser.add_argument('--prepare',action='store_true',help='Create/hash fixed inputs only; does not measure algorithms.')
    parser.add_argument('--run',action='store_true',help='Measure the entire exact reviewed grid once.')
    parser.add_argument('--fixtures')
    parser.add_argument('--output')
    parser.add_argument('--native-python')
    parser.add_argument('--reference-python')
    parser.add_argument('--native-wheel')
    parser.add_argument('--native-build-manifest')
    parser.add_argument('--ct-plan',help='Optional append-only public CT plan; never changes the original58 cases.')
    parser.add_argument('--approved-ct-plan-sha256')
    parser.add_argument('--public-ct-directory',help='External source artifact directory; never copied into the source repository.')
    parser.add_argument('--ct-protocol')
    parser.add_argument('--ct-reference')
    parser.add_argument('--ct-crop-reference')
    parser.add_argument('--dimension-binding')
    parser.add_argument('--approved-dimension-binding-sha256')
    parser.add_argument('--worker',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        request=json.loads(Path(args.worker).read_text(encoding='utf-8'))
        try:
            result=worker(request)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            try:
                peak=peak_resident_bytes()
            except Exception:
                peak=None
            result=dict(status='failed',error=f'{type(exc).__name__}: {exc}',peak_memory_bytes=peak)
        print(json.dumps(clean(result),allow_nan=False,sort_keys=True),flush=True)
        # Observe the kernel HWM after the numerical result has been serialized
        # and transported, rather than underreporting that final process stage.
        try:
            footer_peak=peak_resident_bytes()
        except Exception:
            footer_peak=None
        print(json.dumps(dict(worker_peak_after_transport_bytes=footer_peak)),flush=True)
        return 0 if result['status']=='ok' else 1
    if args.prepare==args.run:
        parser.error('Choose exactly one of --prepare or --run.')
    if not args.fixtures:
        parser.error('--fixtures is required.')
    if args.run and not all((args.approved_plan_sha256,args.output,args.native_python,args.reference_python,args.native_wheel,args.native_build_manifest)):
        parser.error('--run requires the exact approved plan SHA, fresh output, both interpreter paths, fixed wheel and build manifest.')
    if args.ct_plan and not args.approved_ct_plan_sha256:
        parser.error('CT prepare/measurement requires its separately reviewed exact supplemental SHA.')
    if args.run and not all((args.dimension_binding,args.approved_dimension_binding_sha256)):
        parser.error('Measurement requires the explicitly approved true2D inactive-axis interface binding; no silent margin adaptation.')
    plan,digest=load_plan(args.plan,args.approved_plan_sha256 if args.run else None)
    if args.prepare:
        ct_plan=ct_sha=None
        if args.ct_plan:
            ct_plan,ct_sha=load_ct_plan(args.ct_plan,digest,args.approved_ct_plan_sha256)
        prepare(plan,digest,args.fixtures,ct_plan,ct_sha,args.public_ct_directory,
            {'protocol':args.ct_protocol,'original_reference':args.ct_reference,'tight_bbox_reference':args.ct_crop_reference})
        print('Prepared fixed inputs; no algorithm timings collected.')
    else:
        run(args,plan,digest)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
