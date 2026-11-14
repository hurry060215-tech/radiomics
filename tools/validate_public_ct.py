"""Independent public PAT1 download/reference preparation; never imports native code.

Raw CT/ROI inputs stay outside every Git repository. The checked-in protocol and
derived numerical records retain CC-BY-NC-3.0 source provenance, not the software
MIT license. Reference generation requires the unmodified public 3.0.1 environment
and refuses existing outputs. Native acceptance is a separate post-review step.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import urllib.request
import gzip
import struct

ROOT=Path(__file__).resolve().parents[1]
PROFILE=ROOT/'golden/public_ct_v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_protocol():
    return json.loads((PROFILE/'protocol.json').read_text(encoding='utf-8'))


def outside_repository(directory):
    path=Path(directory).resolve()
    if any((parent/'.git').exists() for parent in [path,*path.parents]):
        raise ValueError('Noncommercial source CT/ROI must remain outside every Git worktree.')
    return path


def download(directory,protocol):
    directory=outside_repository(directory)
    pending=[]
    for name,record in protocol['source']['files'].items():
        if Path(name).name!=name:
            raise ValueError('Source filenames must be simple fixed names.')
        target=directory/name
        if target.exists():
            data=target.read_bytes()
        else:
            with urllib.request.urlopen(record['url'],timeout=60) as response:
                data=response.read()
        if len(data)!=record['bytes'] or sha(data)!=record['sha256']:
            raise ValueError('Pinned public CT source length/SHA256 mismatch: '+name)
        if not target.exists():pending.append((target,data))
    directory.mkdir(parents=True,exist_ok=True)
    for path,data in pending:
        with path.open('xb') as output:output.write(data)
    return directory


def header(path):
    raw=gzip.decompress(path.read_bytes());value=raw[:348]
    endian='<' if struct.unpack('<i',value[:4])[0]==348 else '>'
    if struct.unpack(endian+'i',value[:4])[0]!=348:
        raise ValueError('Expected frozen NIfTI-1 source header.')
    return dict(shape_xyz=list(struct.unpack(endian+'8h',value[40:56]))[1:4],
        datatype=struct.unpack(endian+'h',value[70:72])[0],bitpix=struct.unpack(endian+'h',value[72:74])[0],
        pixdim=list(struct.unpack(endian+'8f',value[76:108])),
        scl_slope=struct.unpack(endian+'f',value[112:116])[0],scl_intercept=struct.unpack(endian+'f',value[116:120])[0],
        xyzt_units=value[123],qform_code=struct.unpack(endian+'h',value[252:254])[0],
        sform_code=struct.unpack(endian+'h',value[254:256])[0],
        quaternion=list(struct.unpack(endian+'3f',value[256:268])),qoffset=list(struct.unpack(endian+'3f',value[268:280])),
        sform_rows=[list(struct.unpack(endian+'4f',value[index:index+16])) for index in (280,296,312)])


def environment(protocol):
    import numpy as np
    import SimpleITK as sitk
    import radiomics
    required=protocol['reference_environment']
    actual=dict(python=platform.python_version(),pyradiomics=importlib.metadata.version('pyradiomics'),
        numpy=np.__version__,SimpleITK=sitk.Version_VersionString())
    for name in ('python','pyradiomics','numpy','SimpleITK'):
        if actual[name]!=required[name]:raise RuntimeError('Frozen public reference environment mismatch for '+name)
    if radiomics.__version__!='v3.0.1':raise RuntimeError('Only public PyRadiomics v3.0.1 is a reference.')
    package=Path(radiomics.__file__).parent
    actual['pyradiomics_source_revision']=required['pyradiomics_source_revision']
    actual['pyradiomics_files_sha256']={path.name:sha(path.read_bytes()) for path in sorted(package.iterdir()) if path.suffix in ('.py','.pyd','.so')}
    actual['packages']={name:importlib.metadata.version(name) for name in ('numpy','SimpleITK','pyradiomics','scipy','PyWavelets','pykwalify','six','setuptools','wheel')}
    return actual


def geometry(image):
    import numpy as np
    direction=np.asarray(image.GetDirection()).reshape(3,3)
    affine=np.eye(4);affine[:3,:3]=direction*np.asarray(image.GetSpacing());affine[:3,3]=image.GetOrigin()
    affine=np.diag([-1.,-1.,1.,1.])@affine
    return dict(shape_xyz=list(image.GetSize()),ndim=image.GetDimension(),affine_ras_mm=affine.tolist(),
        spacing_mm=list(image.GetSpacing()),origin_lps_mm=list(image.GetOrigin()),direction_lps=list(image.GetDirection()),spatial_unit='millimeter')


def basic_shape(image,mask):
    import numpy as np
    coordinates=np.argwhere(mask)
    indices_xyz=coordinates[:,::-1]
    affine=np.asarray(geometry(image)['affine_ras_mm'])
    centroid=(affine@np.r_[indices_xyz.mean(axis=0),1.])[:3]
    result={'original_shape_basic_VoxelCount':len(coordinates),
        'original_shape_basic_Volume_mm3':float(len(coordinates)*abs(np.linalg.det(affine[:3,:3])))}
    for dimension,axis in enumerate('XYZ'):
        result['original_shape_basic_BoundingBoxMin'+axis+'_index']=int(indices_xyz[:,dimension].min())
        result['original_shape_basic_BoundingBoxSize'+axis+'_voxels']=int(np.ptp(indices_xyz[:,dimension])+1)
        result['original_shape_basic_Centroid'+axis+'_mm']=float(centroid[dimension])
    return result


def features(image,mask,protocol):
    result={}
    for family,names in protocol['features'].items():
        module=importlib.import_module('radiomics.'+family)
        cls=getattr(module,'RadiomicsFirstOrder' if family=='firstorder' else 'Radiomics'+family.upper())
        extractor=cls(image,mask,**protocol['settings'])
        for name in names:extractor.enableFeatureByName(name)
        values=extractor.execute()
        if set(values)!=set(names):raise ValueError('Reference enabled feature set differs from frozen protocol.')
        for name,value in values.items():
            scalar=float(value)
            result['original_'+family+'_'+name]=scalar if math.isfinite(scalar) else 'NaN' if math.isnan(scalar) else 'Inf' if scalar>0 else '-Inf'
    if len(result)!=55:raise ValueError('Exactly 55 explicitly supported scalar features are required.')
    return result


def sparse(matrix,gray_levels):
    import numpy as np
    indices=np.argwhere(matrix!=0)
    if not np.all(matrix==np.rint(matrix)):raise ValueError('Expected exact integer reference count matrix.')
    return dict(row_values=[int(v) for v in gray_levels],column_values=sorted({int(j+1) for _,j in indices}),
        entries=[[int(i+1),int(j+1),int(matrix[i,j])] for i,j in indices])


def matrix_reference(image,mask,protocol):
    import numpy as np
    import SimpleITK as sitk
    from radiomics import cMatrices,imageoperations
    settings=protocol['matrix_case'];start=settings['start_xyz'];size=settings['size_xyz']
    image=sitk.RegionOfInterest(image,size,start);mask=sitk.RegionOfInterest(mask,size,start)
    array=sitk.GetArrayFromImage(image).astype(np.float64);roi=sitk.GetArrayFromImage(mask)==1
    binned,edges=imageoperations.binImage(array,np.where(roi),**protocol['settings'])
    binned=np.asarray(binned,dtype=np.int32);binned[~roi]=0
    levels=np.unique(binned[roi]);ng=int(levels.max());common=(binned,roi)
    glcm,angles=cMatrices.calculate_glcm(*common,np.asarray([1],dtype=int),ng,False,0)
    glcm+=glcm.transpose(0,2,1,3).copy()
    glrlm,run_angles=cMatrices.calculate_glrlm(*common,ng,max(array.shape),False,0)
    ngtdm=cMatrices.calculate_ngtdm(*common,np.asarray([1],dtype=int),ng,False,0)[0]
    return dict(id=settings['id'],geometry=geometry(image),roi_voxels=int(roi.sum()),
        image_f64_le_sha256=sha(array.astype('<f8').tobytes(order='C')),
        mask_u32_le_sha256=sha(roi.astype('<u4').tobytes(order='C')),
        discrete_u32_le_sha256=sha(binned.astype('<u4').tobytes(order='C')),gray_levels=levels.tolist(),bin_edges=edges.tolist(),
        glcm=[dict(offset_xyz=offset[::-1].tolist(),matrix=sparse(glcm[0,:,:,i],levels)) for i,offset in enumerate(angles)],
        glrlm=[dict(offset_xyz=offset[::-1].tolist(),matrix=sparse(glrlm[0,:,:,i],levels)) for i,offset in enumerate(run_angles)],
        glszm=sparse(cMatrices.calculate_glszm(*common,ng,int(roi.sum()),False,0)[0],levels),
        gldm=sparse(cMatrices.calculate_gldm(*common,np.asarray([1],dtype=int),ng,0,False,0)[0],levels),
        ngtdm=[dict(gray_level=int(row[2]),count=int(row[0]),difference_sum=float(row[1])) for row in ngtdm if row[0]>0])


def reference(directory,output,protocol):
    import numpy as np
    import SimpleITK as sitk
    output=Path(output)
    if output.exists():raise FileExistsError('Refusing existing reference output; create a separate review candidate.')
    state=environment(protocol);directory=download(directory,protocol)
    image=sitk.Cast(sitk.ReadImage(str(directory/'image.nii.gz')),sitk.sitkFloat64)
    mask=sitk.ReadImage(str(directory/'mask.nii.gz'))
    if geometry(image)!=geometry(mask):raise ValueError('Pinned source image/mask geometry differs; no correction is applied.')
    array=sitk.GetArrayFromImage(image).astype(np.float64);labels=sitk.GetArrayFromImage(mask)
    if not np.isfinite(array).all() or not np.array_equal(array,np.rint(array)) or not np.array_equal(np.unique(labels),[0,1]):
        raise ValueError('Frozen exact-integer input behavior changed; no implicit rounding/casting is permitted.')
    headers={name:header(directory/name) for name in ('image.nii.gz','mask.nii.gz')}
    if any(value['xyzt_units']&7!=2 or value['scl_slope']!=1.0 or value['scl_intercept']!=0.0 for value in headers.values()):
        raise ValueError('Frozen units/scaling changed.')
    result=dict(schema='radiomics.public_ct_v1.reference.v1',protocol_sha256=sha((PROFILE/'protocol.json').read_bytes()),
        license=protocol['license'],environment=state,source_headers=headers,geometry=geometry(image),
        output_feature_order=protocol['output_feature_order'],
        image_f64_le_sha256=sha(array.astype('<f8').tobytes(order='C')),mask_u32_le_sha256=sha(labels.astype('<u4').tobytes(order='C')),
        rounding_observation=dict(whole_noninteger_voxels=0,roi_noninteger_voxels=0,rounding_applied=False),cases={},
        basic_shape=basic_shape(image,labels==1))
    for case in protocol['cases']:
        roi=labels==1
        if case['intensity_range'] is not None:
            low,high=case['intensity_range'];roi=roi&(array>=low)&(array<=high)
        intensity=sitk.GetImageFromArray(roi.astype(np.uint8));intensity.CopyInformation(image)
        result['cases'][case['id']]=dict(intensity_roi_voxels=int(roi.sum()),morphology_roi_voxels=int((labels==1).sum()),
            intensity_mask_u32_le_sha256=sha(roi.astype('<u4').tobytes(order='C')),features=features(image,intensity,protocol))
    result['matrix_case']=matrix_reference(image,mask,protocol)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes((json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('download','reference'))
    parser.add_argument('--data-dir',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();protocol=load_protocol()
    if args.operation=='download':
        print(download(args.data_dir,protocol));return
    if args.output is None:parser.error('reference requires a fresh --output candidate path')
    value=reference(args.data_dir,args.output,protocol)
    print(json.dumps(dict(output=str(args.output),sha256=sha(args.output.read_bytes()),
        cases=list(value['cases']),feature_counts={k:len(v['features']) for k,v in value['cases'].items()},matrix_roi=value['matrix_case']['roi_voxels']),indent=2))


if __name__=='__main__':main()
