"""ID-aligned external embeddings → hierarchy → patient/tile models and real tile maps."""
import json
import numpy as np
import pandas as pd
import ov_radiomics as rad
from _shared import output_arguments,task_data,write_json,provenance


def main():
    output=output_arguments(__doc__)
    source,features,classes=task_data('binary')
    records=[];values=[];parents={};status={};metadata=[]
    for _,row in source.iterrows():
        slide=row['ID']+'-slide';parents[slide]=row.patient_id;status[slide]='provided'
        for position in range(2):
            tile=slide+'-tile'+str(position)
            records.append(dict(patient_id=row.patient_id,slide_id=slide,tile_id=tile,x0=position*3.,y0=0.,
                width0=4.,height0=4.,patch_width=4,patch_height=4,valid_pixels=[0,0,4,4],
                mpp_x_um=None,mpp_y_um=None,source='synthetic external embedding; source images already decoded elsewhere',qc={},exclusion_reasons=[]))
            values.append([row[name]+position*.01 for name in features])
            metadata.append(dict(ID=tile,patient_id=row.patient_id,slide_id=slide,group=row.group,target=row.target))
    # Reverse only embedding row order. Equality of row counts would not prove
    # alignment; Rust matches the complete explicit tile-ID population.
    permutation=np.arange(len(values))[::-1]
    ids=[records[i]['tile_id'] for i in permutation]
    matrix=np.asarray(values)[permutation]
    np.save(output/'embeddings.npy',matrix)
    write_json(output/'embedding-tile-ids.json',ids)
    bare=dict(tiles=records,features=None,feature_names=[],slide_patients=parents,slide_status=status)
    table=rad.interop.from_arrays(matrix,tile_ids=ids,tiles=bare,feature_names=features)
    np.testing.assert_array_equal(table.features,np.asarray(values))
    write_json(output/'tiles.json',table.to_dict());write_json(output/'adapter-provenance.json',table.provenance)
    aggregate=rad.pathology.aggregate(table,patient_stats=['mean'])
    aggregate.patient_features.save(output/'patient-features.parquet')
    aggregate.slide_features.save(output/'slide-features.csv')
    write_json(output/'aggregation.json',aggregate.report)
    outcomes=source[['patient_id','group','target']].drop_duplicates('patient_id')
    patients=aggregate.patient_features.features.merge(outcomes,left_on='ID',right_on='patient_id',validate='one_to_one')
    patient_features=['patient_mean(slide_tile_mean('+name+'))' for name in features]
    patient=rad.train_models(patients,'target',patient_features,classes=classes,positive_label=classes[1],fit_unit='patient',
        scale='standard',backend='rust',cv=3,inner_cv=2)
    patient.save(output/'patient-model');rad.evaluate(patient,ci_method=None).to_csv(output/'patient-metrics.csv',index=False)
    # A separate tile-unit model supplies actual local scores. It reuses the same
    # patient partition, while its stored prediction unit remains tile.
    tiles=pd.DataFrame(metadata)
    for i,name in enumerate(features):
        tiles[name]=table.features[:,i]
    tile=rad.train_models(tiles,'target',features,classes=classes,positive_label=classes[1],fit_unit='tile',
        scoring_unit='patient',scale='standard',backend='rust',cv=3,inner_cv=2)
    tile.save(output/'tile-model')
    external=tiles.loc[tiles.group!='train'].drop(columns='target')
    local=tile.predict(external)
    write_json(output/'actual-tile-predictions.json',local.to_dict())
    first=external.loc[external.group=='test'].iloc[0]
    canvas=dict(width=7,height=4,origin0=[0.,0.],pixel_size0=[1.,1.])
    stream=rad.pathology.rasterize_predictions(table,local,slide_id=first.slide_id,class_name=classes[1],canvas=canvas,block_size=(3,2))
    stream.save(output/'actual-tile-score-overlay')
    try:
        rad.pathology.rasterize_predictions(table,patient.predict(patients.loc[patients.group!='train'].drop(columns='target')),
            slide_id=first.slide_id,class_name=classes[1],canvas=canvas)
    except ValueError as error:
        write_json(output/'patient-broadcast-rejected.json',dict(error=str(error)))
    else:
        raise AssertionError('Patient probabilities must not be broadcast into local tile maps.')
    provenance(output,'06_embedding_overlay',embedding_rows=len(matrix),patients=len(patients),
        alignment='complete tile ID equality, deliberately reversed embedding rows',
        coordinates='level-0 pixel edges; mpp unknown, no physical units inferred',
        patient_aggregation='tile→slide then equal-slide patient mean',
        overlay='actual predictions from a separately fitted tile-unit model; grouped patient CV',
        optional_wsidata='See docs/interop.md; generation-time geometry snapshot and TableModel tile-ID linkage required. This external-array tutorial does not claim live WSIData runtime validation.')
    print(rad.evaluate(tile,ci_method=None).to_string(index=False));print('Outputs:',output)


if __name__=='__main__':
    main()
