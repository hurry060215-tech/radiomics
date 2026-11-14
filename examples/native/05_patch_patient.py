"""RGB8 PNG → streamed patches/QC/native colors → equal-slide patient modeling."""
import json
import numpy as np
import pandas as pd
import ov_radiomics as rad
from _shared import output_arguments,task_data,write_json,write_rgb_png,provenance,GOLDEN


def main():
    output=output_arguments(__doc__);(output/'images').mkdir()
    cases=json.loads((GOLDEN/'pathology_inputs.json').read_text())
    data,_,classes=task_data('binary')
    rows=[];vectors=[];qc_rows=[];parents={};statuses={}
    names=['rgb_R_mean','rgb_G_mean','rgb_B_mean','hue_resultant_length','tissue_fraction']
    labels=data[['patient_id','target','group']].drop_duplicates('patient_id')
    for patient,group in data.groupby('patient_id',sort=False):
        for index,(_,row) in enumerate(group.iterrows()):
            slide=patient+'-S'+str(index);parents[slide]=patient;statuses[slide]='provided'
            yy,xx=np.indices((8,8))
            base=np.array([120+15*row.signal_a,80+12*row.signal_b,135+10*row.noise_a])
            pixels=np.clip(base+(xx+2*yy)[...,None],0,255).astype(np.uint8)
            path=output/'images'/(slide+'.png');write_rgb_png(path,pixels)
            patches=rad.pathology.tile_image(path,patient_id=patient,slide_id=slide,patch_size=(4,4),
                stride=(4,4),mpp_x_um=.5,mpp_y_um=.5)
            for patch in rad.pathology.qc_patches(patches,config={'max_white_fraction':.8}):
                colors=rad.pathology.color_features(patch,config={'fixed_edges':cases['fixed_edges']})
                rows.append(patch.record)
                vectors.append([colors['channels'][channel]['statistics']['mean'] for channel in ('R','G','B')]+
                    [None if colors['hue'] is None else colors['hue']['resultant_length'],patch.qc['scores']['tissue_fraction']])
                qc_rows.append(dict(tile_id=patch.record['tile_id'],qc=patch.qc,colors=colors))
    # An explicitly excluded extra slide demonstrates that empty results retain
    # their patient identity and never become an all-zero model feature vector.
    path=output/'images/QC-empty.png';write_rgb_png(path,np.full((4,4,3),255,dtype=np.uint8))
    patch=next(rad.pathology.qc_patches(rad.pathology.tile_image(path,patient_id='QC-empty',slide_id='QC-empty',patch_size=(4,4)),config={'max_white_fraction':.8}))
    rows.append(patch.record);vectors.append([None]*len(names));parents['QC-empty']='QC-empty';statuses['QC-empty']='all_excluded'
    qc_rows.append(dict(tile_id=patch.record['tile_id'],qc=patch.qc,colors=None))
    table=rad.interop.from_arrays(np.asarray(vectors,dtype=float),tile_ids=[row['tile_id'] for row in rows],feature_names=names,
        tiles=dict(tiles=rows,features=None,feature_names=[],slide_patients=parents,slide_status=statuses))
    write_json(output/'tiles.json',table.to_dict())
    with (output/'qc-colors.jsonl').open('w',encoding='utf-8') as handle:
        for row in qc_rows:
            handle.write(json.dumps(row,allow_nan=False)+'\n')
    aggregated=rad.pathology.aggregate(table,patient_stats=['mean'])
    write_json(output/'aggregation.json',aggregated.report)
    aggregated.slide_features.save(output/'slide-features.parquet')
    aggregated.patient_features.save(output/'patient-features.csv')
    status=aggregated.patients[['patient_id','status']]
    empty=status.loc[status.patient_id=='QC-empty'].iloc[0]
    assert empty.status!='ok'
    # Explicit inner join removes only the demonstration patient without an
    # outcome. Its empty status remains exported and disclosed in provenance.
    patients=aggregated.patient_features.features.merge(labels,left_on='ID',right_on='patient_id',validate='one_to_one')
    features=['patient_mean(slide_tile_mean('+name+'))' for name in names[:3]]
    result=rad.train_models(patients,'target',features,task='binary',classes=classes,positive_label=classes[1],
        fit_unit='patient',scoring_unit='patient',patient_id='patient_id',cv=3,inner_cv=2,
        selector=rad.FeatureSelector(backend='rust',univariate=None,lasso=False),backend='rust')
    result.save(output/'patient-model')
    rad.evaluate(result,ci_method=None).to_csv(output/'patient-metrics.csv',index=False)
    first_slide=next(iter(parents))
    rad.pathology.rasterize_predictions(table,feature='rgb_R_mean',slide_id=first_slide,
        canvas=dict(width=8,height=8,origin0=[0.,0.],pixel_size0=[1.,1.]),block_size=(4,4)).save(output/'tile-feature-overlay')
    objects=rad.pathology.object_features(np.array([[1,1,0,2],[1,0,0,2],[0,0,0,0],[0,0,0,0]],dtype=np.uint32),
        kind='instance',connectivity=4,patch_mpp_um=(.5,.5),overlapping_tiles=True)
    assert objects['count_scope']=='patch_local'
    write_json(output/'patch-objects.json',objects)
    provenance(output,'05_patch_patient',patients=len(patients),tiles=len(table),
        empty_patients_retained=['QC-empty'],modeling_exclusion='QC-empty has no study outcome; explicitly omitted from identified modeling join',
        aggregation='tile→slide, then equal-slide patient mean',
        overlay='actual tile RGB feature; patient probabilities are never local scores',
        hue='angles remain patch display statistics; only resultant length is aggregated')
    print(rad.evaluate(result,ci_method=None).to_string(index=False));print('Outputs:',output)


if __name__=='__main__':
    main()
