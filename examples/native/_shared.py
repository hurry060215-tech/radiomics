"""Reproducible synthetic tutorial utilities, with no reference-backend imports."""
import argparse
import hashlib
import importlib.util
import json
import struct
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
import ov_radiomics as rad

GOLDEN=Path(__file__).resolve().parents[2]/'golden/python_golden_v1'


def output_arguments(description):
    parser=argparse.ArgumentParser(description=description)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    return args.output


def task_data(task):
    source=json.loads((GOLDEN/'model_inputs.json').read_text())['datasets'][task]
    rows=[]
    for row in source['records']:
        value=row['target'] if task=='regression' or row['target'] is None else source['classes'][row['target']]
        rows.append(dict(ID=row['sample_id'],patient_id=row['patient_id'],lesion_id=row['lesion_id'],
            group=row['cohort'],center=row['center'],target=value,**dict(zip(source['feature_names'],row['values']))))
    return pd.DataFrame(rows),source['feature_names'],source['classes']


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def extraction_config():
    return dict(extraction=dict(profile='python_golden_v1',discretization=dict(method='fixed_width',width=.5),
        aggregation='direction_mean',gldm_alpha=0,voxel_array_shift=0.),labels=[1],preparation=[])


def image_cohort(output):
    """Construct 132 tiny scalar NIfTI pairs from CC0 synthetic latent values."""
    module=importlib.util.spec_from_file_location('tutorial_extract',Path(__file__).with_name('01_extract.py'))
    writer=importlib.util.module_from_spec(module);module.loader.exec_module(writer)
    frame,features,classes=task_data('binary')
    inputs=output/'inputs';inputs.mkdir()
    coordinates=[(x,y,z) for z in range(4) for y in range(5) for x in range(6)]
    records=[]
    for _,row in frame.iterrows():
        name=row['ID']
        image=inputs/(name+'.nii');mask=inputs/(name+'_roi.nii')
        writer.write_nifti(image,[row[features[0]]+(.2*x+.3*y+.1*z)*(1+.03*abs(row[features[1]])) for x,y,z in coordinates])
        writer.write_nifti(mask,[int(1<=x<=4 and 1<=y<=3 and 1<=z<=2) for x,y,z in coordinates])
        records.append(dict(sample_id=name,patient_id=row.patient_id,image=str(image),mask=str(mask),
            modality='synthetic',region='tumor',center=row.center,lesion_id=row.lesion_id))
    pd.DataFrame(records).to_csv(output/'cohort.csv',index=False)
    config=extraction_config();write_json(output/'extract.json',config)
    result=rad.extract(rad.read_cohort(output/'cohort.csv'),params=config,backend='rust',profile='python_golden_v1',n_jobs=2)
    result.save(output/'extraction')
    return result,frame,classes


def provenance(output,tutorial,**extra):
    from ov_radiomics._backend import resolve_backend
    binary=Path(resolve_backend('rust').__file__)
    write_json(output/'provenance.json',dict(tutorial=tutorial,package_version=rad.__version__,
        backend=rad.backend_info(),extension_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        dataset='CC0 original synthetic software acceptance data; no clinical claims',**extra))


def save_patient_roc(result,data,path,model='LR'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=next(run for run in result.to_dict()['models'] if run['name']==model)
    table=rad.PredictionTable(run['predictions']).aggregate_patients().dataframe
    held=table.loc[table.group=='test'].copy()
    truth=data[['patient_id','target']].drop_duplicates('patient_id')
    held=held.merge(truth,on='patient_id',validate='one_to_one')
    positive=result.classes[1]
    rad.plot_roc((held.target==positive).astype(int),held.probability)
    plt.savefig(path,dpi=140,bbox_inches='tight');plt.close('all')


def write_rgb_png(path,array):
    """Small synthetic RGB8 input writer; decoding is exercised in the Rust core."""
    array=np.asarray(array,dtype=np.uint8)
    height,width,_=array.shape
    def chunk(name,data):
        return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data)&0xffffffff)
    raw=b''.join(b'\0'+row.tobytes() for row in array)
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))+
        chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))


def stack_example_tables(tables):
    """Concatenate already extracted rows without computing new feature values."""
    payload=tables[0].to_dict()
    sources=[payload['effective_parameters']]
    for table in tables[1:]:
        item=table.to_dict()
        if item['features']!=payload['features']:
            raise ValueError('Example tables must have exactly the same feature definitions/order.')
        for key in ('values','missing'):
            payload['data'][key].extend(item['data'][key])
        payload['data']['rows']+=item['data']['rows']
        payload['samples'].extend(item['samples']);payload['quality'].extend(item['quality'])
        sources.append(item['effective_parameters'])
    payload['effective_parameters']=dict(operation='example_row_concatenation',sources=sources)
    return rad.FeatureTable(payload)
