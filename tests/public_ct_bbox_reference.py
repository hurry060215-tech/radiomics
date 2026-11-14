"""Generate a fresh independent PAT1 tight-bbox benchmark reference (no native)."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('frozen_ct_reference',ROOT/'tools/validate_public_ct.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Create a fresh bbox reference candidate; no overwrite.')
    protocol=reference.load_protocol();environment=reference.environment(protocol)
    directory=reference.download(args.data_dir,protocol)
    import numpy as np
    import SimpleITK as sitk
    image=sitk.Cast(sitk.ReadImage(str(directory/'image.nii.gz')),sitk.sitkFloat64)
    mask=sitk.ReadImage(str(directory/'mask.nii.gz'))
    start=[51,51,17];size=[100,99,26]
    cropped=sitk.RegionOfInterest(image,size,start);cropped_mask=sitk.RegionOfInterest(mask,size,start)
    array=sitk.GetArrayFromImage(cropped).astype('<f8');labels=sitk.GetArrayFromImage(cropped_mask).astype('<u4')
    values=reference.features(cropped,cropped_mask,protocol)
    original=json.loads((reference.PROFILE/'reference.json').read_text(encoding='utf-8'))
    expected=original['cases']['PAT1_original_3d_bin25']['features']
    differences=[]
    for name,value in values.items():
        error=abs(value-expected[name]);tolerance=1e-10+1e-8*abs(expected[name])
        differences.append(dict(feature=name,full_reference=expected[name],cropped_reference=value,absolute_error=error,allowed_error=tolerance,passed=error<=tolerance))
    if not all(row['passed'] for row in differences):raise ValueError('Cropping changed a frozen full-ROI scalar outside the existing tolerance.')
    output=dict(schema='radiomics.public_ct_v1.tight_bbox_reference.v1',
        protocol_sha256=reference.sha((reference.PROFILE/'protocol.json').read_bytes()),
        full_reference_sha256=reference.sha((reference.PROFILE/'reference.json').read_bytes()),
        license=protocol['license'],environment=environment,
        crop=dict(start_xyz=start,size_xyz=size,end_xyz_exclusive=[151,150,43],margin_voxels=[0,0,0]),
        preparation=dict(rounding=None,resampling=None,clipping=None,normalization=None,resegmentation=None),
        settings=protocol['settings'],output_feature_order=protocol['output_feature_order'],
        geometry=reference.geometry(cropped),image_f64_le_sha256=reference.sha(array.tobytes(order='C')),
        mask_u32_le_sha256=reference.sha(labels.tobytes(order='C')),roi_voxels=int((labels==1).sum()),
        features=values,basic_shape=reference.basic_shape(cropped,labels==1),full_vs_crop=differences,
        index_coordinates='BoundingBoxMin indices are local to the cropped grid and reset to zero; size, world centroid and physical volume retain original ROI meaning. No NIfTI reserialization or affine quantization occurs.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes((json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))
    print(json.dumps(dict(output=str(args.output),sha256=reference.sha(args.output.read_bytes()),features=len(values),roi=output['roi_voxels']),indent=2))


if __name__=='__main__':main()
