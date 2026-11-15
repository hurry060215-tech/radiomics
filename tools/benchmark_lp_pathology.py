"""Declared RGB8 front end; frozen independent specification or public Rust API."""
import hashlib
import importlib.util
from pathlib import Path
import struct

from benchmark_lp_learning import clean,write_json


def specification(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def tile_identifier(record,x,y):
    digest=hashlib.sha256();digest.update(b'radiomics-pixel-tile-v1')
    for value in (record['patient_id'],record['slide_id'],record['source']):
        raw=value.encode('utf-8');digest.update(struct.pack('<Q',len(raw)));digest.update(raw)
    for value in (0.,0.,1.,1.):
        digest.update(struct.pack('<d',value))
    for value in (x,y,32,32):
        digest.update(struct.pack('<Q',value))
    return record['slide_id']+':'+digest.hexdigest()


def pixel_frontend(context,input_kind,backend,stage):
    import numpy as np
    controls=context['plan']['pathology'];qc_config=context['pathology_config']['qc_config']
    color_config={'fixed_edges':context['pathology_config']['fixed_edges']}
    tiles=[];vectors=[];details=[];parents={};statuses={}
    names=controls['colors']['generic_tile_matrix_order']
    with stage('decode_tile_qc_color'):
        for image_record in context['images']:
            patient,slide=image_record['patient_id'],image_record['slide_id'];parents[slide]=patient
            if backend=='rust':
                rad=context['rad']
                image=(rad.pathology.read_image(context['fixtures']/image_record['png']) if input_kind=='png'
                    else rad.pathology.from_array(context['decoded'][image_record['npy']]))
                patches=rad.pathology.tile_image(image,patient_id=patient,slide_id=slide,source=image_record['source'],
                    patch_size=(32,32),stride=(24,24),edge='pad',pad_value=0,origin0=(0.,0.),pixel_size0=(1.,1.))
                patches=rad.pathology.qc_patches(patches,config=qc_config)
                for patch in patches:
                    qc=patch.qc;record=patch.record
                    colors=rad.pathology.color_features(patch,config=color_config) if qc['accepted'] else None
                    tiles.append(record);vectors.append(vector(colors,qc));details.append(dict(tile_id=record['tile_id'],qc=qc,colors=colors))
            else:
                spec=context['pixel_spec']
                if input_kind=='png':
                    from PIL import Image
                    with Image.open(context['fixtures']/image_record['png']) as source:
                        if source.mode!='RGB':
                            raise ValueError('The reference PNG must retain explicit RGB8 mode.')
                        pixels=np.asarray(source).copy()
                else:
                    pixels=context['decoded'][image_record['npy']]
                height,width=pixels.shape[:2]
                for geometry in spec.tile_positions(width,height,[32,32],[24,24],'pad'):
                    x,y=int(geometry['x0']),int(geometry['y0']);_,_,w,h=geometry['valid_pixels']
                    patch=np.zeros((32,32,3),dtype=np.uint8);patch[:h,:w]=pixels[y:y+h,x:x+w]
                    raw=patch.reshape(-1,3).tolist();qc=spec.qc(32,32,raw,geometry['valid_pixels'],qc_config)
                    record=dict(geometry,patient_id=patient,slide_id=slide,tile_id=tile_identifier(image_record,x,y),
                        source=image_record['source'],mpp_x_um=None,mpp_y_um=None,
                        qc={key:value for key,value in qc['scores'].items() if value is not None},exclusion_reasons=qc['exclusion_reasons'])
                    colors=spec.colors(32,32,raw,geometry['valid_pixels'],color_config['fixed_edges']) if qc['accepted'] else None
                    tiles.append(record);vectors.append(vector(colors,qc));details.append(dict(tile_id=record['tile_id'],qc=qc,colors=colors))
            statuses[slide]='provided' if any(not row['exclusion_reasons'] for row in tiles if row['slide_id']==slide) else 'all_excluded'
    return dict(tiles=tiles,vectors=vectors,details=details,feature_names=names,slide_patients=parents,slide_status=statuses)


def vector(colors,qc):
    if colors is None:
        return [None]*5
    return [colors['channels'][name]['statistics']['mean'] for name in ('R','G','B')]+[
        colors['hue']['resultant_length'] if colors['hue'] is not None else None,qc['scores']['tissue_fraction']]


def ordered_colors(colors,controls):
    if colors is None:
        return None
    channels={}
    for name in controls['colors']['channels_in_output_order']:
        value=colors['channels'][name];statistics=value['statistics']
        channels[name]=dict(statistics=None if statistics is None else
            {key:statistics[key] for key in controls['colors']['channel_statistics_order']},histogram=value['histogram'])
    hue=colors['hue']
    if hue is not None:
        hue={key:hue[key] for key in ('count','mean_degrees','resultant_length','circular_variance','histogram')}
    return dict(status=colors['status'],channels=channels,hue=hue)


def align_aggregate(context,pixels,backend,stage):
    import numpy as np
    with stage('id_alignment_aggregation'):
        ids=[row['tile_id'] for row in pixels['tiles']]
        reversed_ids=list(reversed(ids));reversed_values=list(reversed(pixels['vectors']))
        if backend=='rust':
            rad=context['rad']
            table=rad.interop.from_arrays(np.asarray(reversed_values,dtype=float),tile_ids=reversed_ids,
                feature_names=pixels['feature_names'],tiles=dict(tiles=pixels['tiles'],features=None,feature_names=[],
                    slide_status=pixels['slide_status'],slide_patients=pixels['slide_patients']))
            hierarchy=rad.pathology.aggregate(table,patient_stats=['mean'])
            canonical=table.to_dict();report=hierarchy.report
        else:
            spec=context['aggregate_spec']
            aligned=spec.align_embeddings(ids,reversed_ids,reversed_values)
            report=spec.aggregate(pixels['tiles'],aligned,pixels['feature_names'],pixels['slide_patients'],patient_stats=['mean'])
            canonical=dict(tiles=pixels['tiles'],feature_names=pixels['feature_names'],slide_status=pixels['slide_status'],
                slide_patients=pixels['slide_patients'],features=dict(rows=len(aligned),cols=5,
                    values=[0. if value is None else value for row in aligned for value in row],
                    missing=[value is None for row in aligned for value in row]))
    return canonical,report


def overlays(context,table,backend,stage):
    import numpy as np
    controls=context['plan']['pathology']['overlays'];results={}
    with stage('actual_tile_overlay_compute'):
        for slide in controls['slides']:
            if backend=='rust':
                blocks=list(context['rad'].pathology.rasterize_predictions(table,slide_id=slide,feature='rgb_R_mean',
                    canvas=controls['canvas'],block_size=controls['block_size_wh'],clip=False))
            else:
                selected=[(i,row) for i,row in enumerate(table['tiles']) if row['slide_id']==slide]
                tiles=[row for _,row in selected]
                eligible=[(i,row) for i,row in selected if not row['exclusion_reasons']]
                ids=[row['tile_id'] for _,row in eligible]
                scores=[table['features']['values'][i*5] for i,_ in eligible]
                blocks=context['aggregate_spec'].raster_blocks(tiles,ids,scores,controls['canvas'],controls['block_size_wh'])
                for block in blocks:
                    shape=(block['height'],block['width'])
                    block['values']=np.asarray([float(value) for value in block['values']]).reshape(shape)
                    block['coverage']=np.asarray(block['coverage'],dtype=np.uint64).reshape(shape)
                    block['alpha']=np.asarray(block['alpha'],dtype=np.uint8).reshape(shape)
                    block['missing']=block['coverage']==0
            results[slide]=blocks
    return results


def export_frontend(context,pixels,table,report,blocks,output,stage):
    import json
    import numpy as np
    import pandas as pd
    output=Path(output)
    with stage('frontend_exports'):
        write_json(output/'tile-table.json',table)
        write_json(output/'aggregation.json',report)
        with (output/'qc-colors.jsonl').open('x',encoding='utf-8',newline='\n') as handle:
            for row in pixels['details']:
                row=dict(row,colors=ordered_colors(row['colors'],context['plan']['pathology']))
                handle.write(json.dumps(clean(row),allow_nan=False)+'\n')
        rows=[];data=table['features']
        for i,tile in enumerate(table['tiles']):
            rows.append(dict(patient_id=tile['patient_id'],slide_id=tile['slide_id'],tile_id=tile['tile_id'],
                x0=tile['x0'],y0=tile['y0'],width0=tile['width0'],height0=tile['height0'],
                **{name:float('nan') if data['missing'][i*5+j] else data['values'][i*5+j] for j,name in enumerate(table['feature_names'])}))
        pd.DataFrame(rows).to_csv(output/'tile-features.csv',index=False)
        for level,identifier in [('slides','slide_id'),('patients','patient_id')]:
            columns=report['slide_feature_names'] if level=='slides' else report['patient_feature_names']
            records=[dict(ID=row[identifier],patient_id=row['patient_id'],status=row['status'],
                **{name:row['features'][name] for name in columns}) for row in report[level]]
            pd.DataFrame(records,columns=['ID','patient_id','status']+columns).to_csv(output/(level+'.csv'),index=False)
        for slide,parts in blocks.items():
            directory=output/'overlays'/slide;directory.mkdir(parents=True)
            entries=[]
            for block in parts:
                name=f"block-{block['y']:012}-{block['x']:012}.npz"
                np.savez_compressed(directory/name,**{key:block[key] for key in ('values','coverage','missing','alpha')})
                entries.append({key:value for key,value in block.items() if key not in ('values','coverage','missing','alpha')}|dict(path=name))
            write_json(directory/'manifest.json',dict(canvas=context['plan']['pathology']['overlays']['canvas'],blocks=entries,
                score='actual_tile_rgb_R_mean',coordinate_space='level0_pixel_edges'))
        write_json(output/'manifest.json',dict(schema='radiomics.benchmark.v2.pathology_frontend_outputs.v1',
            sample_unit='tile',coordinate_space='level0_pixel_edges',physical_scale='unknown; pixel units only',
            tile_feature_names=table['feature_names'],slide_feature_names=report['slide_feature_names'],
            patient_feature_names=report['patient_feature_names'],
            empty_patients=[row['patient_id'] for row in report['patients'] if row['status']!='ok'],
            model_training_included=False,source_specs=context['plan']['source_files_sha256']))


def run_frontend(context,input_kind,backend,output,stage):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    pixels=pixel_frontend(context,input_kind,backend,stage)
    table,report=align_aggregate(context,pixels,backend,stage)
    blocks=overlays(context,table,backend,stage)
    export_frontend(context,pixels,table,report,blocks,output,stage)
    def comparison():
        return clean(dict(tile_table=table,QC_colors=pixels['details'],hierarchy=report,overlays=blocks))
    return dict(comparison=comparison,hierarchy=report,table=table)


def patient_model_dataset(report,labels,feature_names):
    lookup={row['patient_id']:row for row in report['patients']}
    records=[]
    for metadata in labels['records']:
        row=lookup[metadata['patient_id']]
        if row['status']!='ok' or any(row['features'][name] is None for name in feature_names):
            raise ValueError('An outcome-bearing pathology patient unexpectedly has no accepted feature data.')
        records.append(dict(metadata,values=[row['features'][name] for name in feature_names]))
    if len(records)!=12 or {row['patient_id'] for row in report['patients']}-{row['patient_id'] for row in records} != {'QC-empty'}:
        raise ValueError('Only the predeclared no-outcome QC sentinel may be absent from modeling.')
    return dict(feature_names=feature_names,classes=labels['classes'],records=records)
