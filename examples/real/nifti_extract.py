"""Real NIfTI+ROI cohort -> Rust features and actual input ROI previews. No outcomes invented."""
import argparse
from pathlib import Path
import numpy as np
import ov_radiomics as rad
from _common import output_directory,save_feature_table,sha256,write_json


def preview(image_path,mask_path,labels,path,title,window=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    image=rad.image.read_nifti(image_path);mask=rad.mask.read_nifti(mask_path)
    # Geometry was validated in the successful extraction; retain its full
    # affine/units. NumPy here chooses a display plane, not a feature algorithm.
    selected=np.isin(mask.array,labels);values=image.array
    if values.shape!=selected.shape:raise ValueError('Preview image/mask shapes differ.')
    if values.ndim==3:
        z=int(np.argmax(selected.sum(axis=(1,2))));values=values[z];selected=selected[z]
    else:z=None
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    settings={} if window is None else dict(vmin=window[0],vmax=window[1])
    for axis in axes:axis.imshow(values,origin='lower',cmap='gray',**settings);axis.set(xlabel='Voxel x index',ylabel='Voxel y index')
    axes[0].set_title(title+' input image');axes[1].set_title('Actual selected ROI'+(' / axial z='+str(z) if z is not None else ''))
    if selected.any():axes[1].contour(selected.astype(float),levels=[.5],colors=['#ff4b4b'],linewidths=.8)
    fig.suptitle('Voxel-index display; affine retained separately. Window is display-only.',fontsize=9)
    fig.savefig(path,dpi=135);plt.close(fig)
    return dict(image_geometry=image.geometry,mask_geometry=mask.geometry,display_axial_z=z,display_window=window,display_axes='voxel indices, not RAS-reoriented anatomical view')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',type=Path,required=True)
    bins=parser.add_mutually_exclusive_group(required=True);bins.add_argument('--bin-width',type=float);bins.add_argument('--bin-count',type=int)
    parser.add_argument('--labels',type=int,nargs='+',default=[1],help='Explicit selected label set; per-row roi_label overrides this set.')
    parser.add_argument('--features',choices=['basic','full'],default='basic',help='basic: firstorder16+shape_basic11; full: all55+shape_basic11.')
    parser.add_argument('--n-jobs',type=int,default=1);parser.add_argument('--cache',type=Path)
    parser.add_argument('--preview-samples',type=int,default=3);parser.add_argument('--preview-window',type=float,nargs=2)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.n_jobs<1 or args.preview_samples<0:parser.error('n-jobs must be positive and preview-samples nonnegative.')
    if args.bin_width is not None and (not np.isfinite(args.bin_width) or args.bin_width<=0):parser.error('bin-width must be finite and positive.')
    if args.bin_count is not None and args.bin_count<1:parser.error('bin-count must be positive.')
    if any(label<=0 for label in args.labels) or len(set(args.labels))!=len(args.labels):parser.error('labels must be distinct positive integers.')
    if args.preview_window and args.preview_window[0]>=args.preview_window[1]:parser.error('preview window must be ordered.')
    samples=rad.read_cohort(args.cohort)
    discretization=dict(method='fixed_width',width=args.bin_width) if args.bin_width is not None else dict(method='fixed_count',count=args.bin_count)
    config=dict(extraction=dict(profile='python_golden_v1',discretization=discretization,aggregation='direction_mean',gldm_alpha=0,voxel_array_shift=0.),
        labels=args.labels,preparation=[],families=['firstorder','shape_basic'] if args.features=='basic' else None)
    output=output_directory(args.output)
    result=rad.extract(samples,params=config,backend='rust',profile='python_golden_v1',n_jobs=args.n_jobs,cache=args.cache,on_error='record')
    result.save(output/'extraction')
    result.diagnostics.to_csv(output/'diagnostics.csv',index=False)
    failures=result.diagnostics.loc[~result.diagnostics.ok.astype(bool)];failures.to_csv(output/'failures.csv',index=False)
    table=result.feature_table
    if table is not None:save_feature_table(table,output/'features.csv')
    else:samples[['ID','patient_id']].rename(columns={'ID':'sample_id'}).to_csv(output/'metadata.csv',index=False)
    if table is not None:table.metadata.to_csv(output/'metadata.csv',index=False)
    previews=[];(output/'preview').mkdir()
    successful=set(result.diagnostics.loc[result.diagnostics.ok.astype(bool),'ID'])
    for index,row in enumerate(samples.loc[samples.ID.isin(successful)].head(args.preview_samples).to_dict('records')):
        import pandas as pd
        labels=args.labels if row.get('roi_label') is None or pd.isna(row.get('roi_label')) else [int(row['roi_label'])]
        name=f'sample-{index:03}.png';geometry=preview(row['image'],row['mask'],labels,output/'preview'/name,str(row['ID']),args.preview_window)
        previews.append(dict(sample_id=str(row['ID']),labels=labels,path='preview/'+name,**geometry))
    write_json(output/'provenance.json',dict(schema='radiomics.real.nifti.v1',cohort=str(args.cohort.resolve()),cohort_sha256=sha256(args.cohort),
        backend='rust',package_version=rad.__version__,implementation_sha256=result.config['implementation_sha256'],effective_config=config,
        feature_scope=args.features,feature_count=0 if table is None else len(table.definitions),previews=previews,
        input_content_hashes=[dict(sample_id=row['metadata']['sample_id'],sha256=row['input_sha256']) for row in result.config['records']],
        status=result.config['status'],failed_samples=len(failures),outcomes='Not read, generated or inferred. Supply real clinical/outcome tables separately.',
        plot='Actual input image/mask; display window changes no extraction values. No resampling/normalization/clipping in this entry point.'))
    print('features:',output/'features.csv');print('metadata:',output/'metadata.csv');print('failed_samples:',len(failures));print('previews:',output/'preview')
    return 2 if len(failures) else 0


if __name__=='__main__':raise SystemExit(main())
