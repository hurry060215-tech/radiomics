"""Anisotropic peritumoral ROI, keyed multimodal/Delta tables and existing habitats."""
import numpy as np
import pandas as pd
import ov_radiomics as rad
from _shared import output_arguments,extraction_config,stack_example_tables,write_json,provenance


def main():
    output=output_arguments(__doc__)
    yy,xx=np.indices((9,9))
    affine=[[1.,0.,0.,10.],[0.,2.,0.,-20.],[0.,0.,1.,3.],[0.,0.,0.,1.]]
    config=extraction_config()['extraction']
    sources={name:[] for name in ('CT_tumor','CT_ring','MR_tumor','MR_ring')}
    for timepoint,change in [('T0',0.),('T1',.75)]:
        for modality in ('CT','MR'):
            # These are deliberately synthetic, already aligned images. No
            # registration or modality-specific intensity preset is inferred.
            image=rad.image.from_array(2.+xx+3.*yy+change+(xx*yy*.01 if modality=='MR' else 0.),
                affine=affine,spatial_unit='mm')
            tumor=rad.mask.from_array(((xx>=3)&(xx<=5)&(yy>=3)&(yy<=5)).astype(np.uint32),geometry=image.geometry)
            organ=rad.mask.from_array(np.ones((9,9),dtype=np.uint32),geometry=image.geometry)
            ring=rad.mask.peritumoral_ring(tumor,inner_mm=0.,outer_mm=3.,organ_mask=organ)
            assert not np.any(tumor.array.astype(bool)&ring.array.astype(bool))
            for region,mask in [('tumor',tumor),('ring',ring)]:
                sample=rad.ImageSample(dict(sample_id=modality+'-'+region+'-'+timepoint,patient_id='P1',lesion_id='L1',
                    modality=modality,region=region,timepoint=timepoint,center='A'),image,mask)
                sources[modality+'_'+region].append(sample.extract(config))
            if modality=='CT' and timepoint=='T0':
                # Two given synthetic labels demonstrate reuse of an externally
                # available habitat mask, not a segmentation algorithm.
                given=np.where(tumor.array>0,np.where(xx<5,1,2),0).astype(np.uint32)
                habitat_mask=rad.mask.from_array(given,geometry=image.geometry)
                np.save(output/'provided-habitat-mask.npy',given)
                for label in (1,2):
                    roi=rad.mask.select_labels(habitat_mask,[label])
                    rad.ImageSample(dict(sample_id='provided-habitat-'+str(label),patient_id='P1'),image,roi).extract(config).save(output/('provided-habitat-'+str(label)+'.csv'))
    tables={name:stack_example_tables(items) for name,items in sources.items()}
    merged=rad.research.concat(tables,key_fields=['patient_id','lesion_id','timepoint'],
        varying_fields=['modality','region'],already_registered=True)
    merged.save(output/'multimodal-multiregion.parquet')
    write_json(output/'concat-provenance.json',merged.provenance)
    change=rad.research.delta(tables['CT_tumor'],baseline='T0',followup='T1',zero_denominator='missing')
    change.save(output/'delta.csv');write_json(output/'delta-pairing.json',change.provenance)
    # Basic habitat fitting accepts an existing complete intensity/feature matrix.
    # This model is trained on these observations only; labels are local to it.
    training=pd.DataFrame([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[8.,8.],[9.,8.],[8.,9.],[9.,9.]],
        columns=['intensity_a','intensity_b'],index=['voxel-'+str(i) for i in range(8)])
    habitat=rad.habitat.fit(training,clusters=2,initial_centers=[[0.,0.],[8.,8.]])
    habitat.save(output/'local-habitat.json')
    restored=rad.habitat.HabitatModel.load(output/'local-habitat.json')
    pd.testing.assert_frame_equal(habitat.predict(training),restored.predict(training))
    habitat.predict(training).to_csv(output/'local-habitat-labels.csv',index=False)
    habitat.centers_.to_csv(output/'local-habitat-centers.csv',index=False)
    provenance(output,'04_regions_delta_habitat',registration='caller-declared synthetic common grid; none performed',
        ring='Euclidean mm distance with x=1mm/y=2mm anisotropy and organ intersection',
        delta='explicit paired patient/lesion/modality/region; relative change is a fraction; zero denominator policy=missing',
        habitat='single local fit; cluster numbers do not imply shared histologic tissue types')
    print('Outputs:',output)


if __name__=='__main__':
    main()
