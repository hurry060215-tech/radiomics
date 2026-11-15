"""Existing ordinary patch files+explicit mapping -> Rust QC/features/hierarchy/tile-feature maps."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import ov_radiomics as rad
from _common import output_directory,plot_overlay,plot_patch_qc,read_rows,save_feature_table,sha256,write_json


FEATURES=['rgb_R_mean','rgb_G_mean','rgb_B_mean','lab_L_mean','hue_resultant_length','qc_tissue_fraction','qc_laplacian_variance']


def default_color_config():
    # Display/feature histogram choices are explicit example parameters, not a
    # stain model or a guessed modality-dependent extraction setting.
    return {'fixed_edges':{'R':list(np.linspace(0,256,17)),'G':list(np.linspace(0,256,17)),'B':list(np.linspace(0,256,17)),
        'S':list(np.linspace(0,1,9)),'V':list(np.linspace(0,1,9)),'H':list(np.linspace(0,360,13)),
        'L':list(np.linspace(0,100,11)),'a':list(np.linspace(-128,128,17)),'b':list(np.linspace(-128,128,17))}}


def optional_number(row,key):
    return None if row.get(key,'').strip()=='' else float(row[key])


def integer(row,key,default):
    if row.get(key,'').strip()=='':return default
    value=float(row[key])
    if not math.isfinite(value) or value!=math.floor(value):raise ValueError(key+' must be an exact integer.')
    return int(value)


def load_manifest(path,coordinate_space):
    rows=read_rows(path,['path','patient_id','slide_id','tile_id'])
    if len({row['tile_id'] for row in rows})!=len(rows):raise ValueError('tile_id must be unique; row counts cannot establish alignment.')
    parents={}
    for row in rows:
        if row['slide_id'] in parents and parents[row['slide_id']]!=row['patient_id']:raise ValueError('A slide has conflicting patient/group parents.')
        parents[row['slide_id']]=row['patient_id']
        row['path']=str((Path(path).parent/row['path']).resolve())
        if coordinate_space=='level0' and any(row.get(key,'').strip()=='' for key in ('x0','y0','width0','height0')):
            raise ValueError('Original level0 maps require supplied x0,y0,width0,height0; no row-position coordinates are invented.')
    if coordinate_space=='image-local' and len(parents)!=len(rows):
        raise ValueError('image-local requires one complete source-image file per slide/group; it cannot place unrelated patches into a fabricated slide.')
    if coordinate_space=='image-local' and any(row.get(key,'').strip() for row in rows for key in ('x0','y0','width0','height0')):
        raise ValueError('image-local cannot silently override supplied original coordinates; use level0 mode with those records.')
    return rows,parents


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--coordinate-space',choices=['level0','image-local'],default='level0')
    parser.add_argument('--identity-scope',choices=['patient','source-image'],default='patient')
    parser.add_argument('--qc-config',type=Path);parser.add_argument('--color-config',type=Path)
    parser.add_argument('--canvases',type=Path,help='JSON object slide_id -> native Canvas; without this use only the extent of authoritative supplied tiles.')
    parser.add_argument('--overlay-feature',choices=FEATURES,default='rgb_R_mean')
    parser.add_argument('--overlay-pixel-size',type=float,default=1.,help='Output display pixel size in source/level0 pixels, never MPP.')
    args=parser.parse_args()
    if not math.isfinite(args.overlay_pixel_size) or args.overlay_pixel_size<=0:parser.error('overlay-pixel-size must be positive finite.')
    rows,parents=load_manifest(args.manifest,args.coordinate_space)
    qc_config={} if args.qc_config is None else json.loads(args.qc_config.read_text(encoding='utf-8'))
    color_config=default_color_config() if args.color_config is None else json.loads(args.color_config.read_text(encoding='utf-8'))
    canvases={} if args.canvases is None else json.loads(args.canvases.read_text(encoding='utf-8'))
    if set(canvases)-set(parents):raise ValueError('Canvas refers to unknown slide IDs.')
    output=output_directory(args.output);records=[];vectors=[];failures=[];inputs=[];qc_rows=[];(output/'preview').mkdir()
    for row in rows:
        try:
            source=Path(row['path']);digest=sha256(source);image=rad.pathology.read_image(source);pixels=image.to_dict()
            width,height=pixels['width'],pixels['height']
            if args.coordinate_space=='image-local':x0,y0,width0,height0=0.,0.,float(width),float(height)
            else:x0,y0,width0,height0=[float(row[key]) for key in ('x0','y0','width0','height0')]
            valid=[integer(row,'valid_x',0),integer(row,'valid_y',0),integer(row,'valid_width',width),integer(row,'valid_height',height)]
            record=dict(patient_id=row['patient_id'],slide_id=row['slide_id'],tile_id=row['tile_id'],x0=x0,y0=y0,width0=width0,height0=height0,
                patch_width=width,patch_height=height,valid_pixels=valid,mpp_x_um=optional_number(row,'mpp_x_um'),mpp_y_um=optional_number(row,'mpp_y_um'),
                source=str(source),qc={},exclusion_reasons=[])
            patch=rad.pathology.Patch(dict(record=record,image=pixels));patch=next(rad.pathology.qc_patches([patch],config=qc_config))
            colors=rad.pathology.color_features(patch,config=color_config)
            if len(records)<3:plot_patch_qc(patch,output/'preview'/f'patch-{len(records):03}.png')
            def mean(channel):
                stats=colors['channels'][channel]['statistics'];return None if stats is None else stats['mean']
            vector=[mean('R'),mean('G'),mean('B'),mean('L'),None if colors['hue'] is None else colors['hue']['resultant_length'],
                patch.qc['scores']['tissue_fraction'],patch.qc['scores']['laplacian_variance']]
            if sha256(source)!=digest:raise ValueError('Patch file changed during processing.')
            records.append(patch.record);vectors.append(vector)
            qc_rows.append(dict(tile_id=row['tile_id'],patient_id=row['patient_id'],slide_id=row['slide_id'],**{k:v for k,v in patch.qc.items() if k!='tissue_mask'}))
            inputs.append(dict(tile_id=row['tile_id'],path=str(source),sha256=digest,decoded_width=width,decoded_height=height))
        except Exception as error:
            failures.append(dict(tile_id=row['tile_id'],patient_id=row['patient_id'],slide_id=row['slide_id'],path=row['path'],error=str(error)))
    statuses={}
    for slide in parents:
        tiles=[row for row in records if row['slide_id']==slide]
        statuses[slide]='failed' if not tiles else 'all_excluded' if all(row['exclusion_reasons'] for row in tiles) else 'partial' if any(row['slide_id']==slide for row in failures) else 'provided'
    geometry=dict(tiles=records,features=None,feature_names=[],slide_patients=parents,slide_status=statuses)
    write_json(output/'tiles-geometry.json',geometry)
    pd.DataFrame(failures,columns=['tile_id','patient_id','slide_id','path','error']).to_csv(output/'failures.csv',index=False)
    with (output/'qc.jsonl').open('x',encoding='utf-8',newline='\n') as handle:
        for row in qc_rows:handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    overlays=[]
    if records:
        matrix=np.asarray([[np.nan if value is None else value for value in row] for row in vectors],dtype=float)
        table=rad.interop.from_arrays(matrix,tile_ids=[row['tile_id'] for row in records],tiles=geometry,feature_names=FEATURES,source=str(args.manifest.resolve()))
        write_json(output/'tiles.json',table.to_dict())
        tile_metadata=pd.DataFrame(records);tile_metadata.to_csv(output/'tiles-metadata.csv',index=False)
        # Canonical table export reuses the same native type/serialization logic.
        metadata=[dict(sample_id=row['tile_id'],patient_id=row['patient_id'],slide_id=row['slide_id'],lesion_id=None,modality=None,region=None,timepoint=None,center=None) for row in records]
        canonical=rad.FeatureTable(dict(data=table.to_dict()['features'],samples=metadata,features=[dict(name=name,definition='Native ordinary patch QC/color statistic',unit=None) for name in FEATURES],
            effective_parameters=dict(qc=qc_config,color=color_config,identity_scope=args.identity_scope,coordinate_space=args.coordinate_space),quality=[]))
        save_feature_table(canonical,output/'tile-features.csv')
        aggregate=rad.pathology.aggregate(table,patient_stats=['mean']);write_json(output/'aggregation.json',aggregate.report)
        save_feature_table(aggregate.slide_features,output/'slide-features.csv')
        group_name='patient-features.csv' if args.identity_scope=='patient' else 'group-features.csv';save_feature_table(aggregate.patient_features,output/group_name)
        for index,slide in enumerate(parents):
            accepted=[row for row in records if row['slide_id']==slide and not row['exclusion_reasons']]
            if not accepted:overlays.append(dict(slide_id=slide,status='no_accepted_tiles'));continue
            canvas=canvases.get(slide);scope='explicit_user_canvas'
            if canvas is None:
                left=min(row['x0'] for row in accepted);top=min(row['y0'] for row in accepted)
                right=max(row['x0']+row['width0'] for row in accepted);bottom=max(row['y0']+row['height0'] for row in accepted);step=args.overlay_pixel_size
                canvas=dict(width=math.ceil((right-left)/step),height=math.ceil((bottom-top)/step),origin0=[left,top],pixel_size0=[step,step]);scope='extent_of_provided_tiles_not_unknown_full_slide'
            if canvas['width']*canvas['height']>4_000_000:
                overlays.append(dict(slide_id=slide,status='preview_too_large',recommendation='Increase --overlay-pixel-size or supply bounded explicit canvas.'));continue
            directory=output/f'overlay-{index:03}'
            rad.pathology.rasterize_predictions(table,feature=args.overlay_feature,slide_id=slide,canvas=canvas,block_size=(256,256)).save(directory)
            plot_overlay(directory,directory.with_suffix('.png'),args.overlay_feature,'Source-image pixels' if args.coordinate_space=='image-local' else 'Level-0 pixels')
            overlays.append(dict(slide_id=slide,status='written',path=directory.name,canvas=canvas,scope=scope))
    else:write_json(output/'tiles.json',geometry)
    write_json(output/'overlays.json',overlays)
    from ov_radiomics import _native
    write_json(output/'provenance.json',dict(schema='radiomics.real.patches.v1',manifest=str(args.manifest.resolve()),manifest_sha256=sha256(args.manifest),
        backend='rust',package_version=rad.__version__,identity_scope=args.identity_scope,coordinate_space=args.coordinate_space,
        implementation_sha256=_native.IMPLEMENTATION_SHA256,
        identity_notice='IDs are caller-supplied; source-image groups are not established clinical patients or original histology slide identities.',
        local_coordinate_notice='image-local means the supplied ordinary source image itself, never inferred WSI placement.',inputs=inputs,qc_config=qc_config,color_config=color_config,
        feature_names=FEATURES,failed_files=len(failures),tile_count=len(records),overlays=overlays,outcomes='No outcome/clinical split is generated; no model training occurs.',
        mpp='Only explicit input MPP accepted; absence remains unknown. Hue angle means are not linear aggregation features.'))
    print('tiles:',output/'tiles.json');print('features:',output/'tile-features.csv');print('hierarchy:',output);print('failed_files:',len(failures))
    return 2 if failures else 0


if __name__=='__main__':raise SystemExit(main())
