"""Independent ML/statistics and pathology workflows under the reviewed v2 plan.

Input preparation is deterministic and does not fit models, compute pathology
features, record new Golden values or collect algorithm timings. Measurement
requires a separately approved plan and immutable artifact/environment binding.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
SOURCE_TEXT_BINDING_SHA256='0a0944f104703e3008a93ad36ec56e5e6d473828fe6890e950c602e1f7f97c9c'


def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON configuration key: '+key)
        result[key]=value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=unique_object)


def file_hash(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as destination:
        json.dump(value,destination,allow_nan=False,sort_keys=True,indent=2)
        destination.write('\n')


def load_plan(path,approved_sha256=None,source_root=ROOT):
    digest=file_hash(path);plan=read_json(path)
    if approved_sha256 is not None and digest!=approved_sha256:
        raise ValueError('Plan bytes differ from the exact main-reviewed SHA256.')
    if plan.get('schema')!='radiomics.benchmark.v2.learning_pathology_plan.v1':
        raise ValueError('Expected the independent learning/pathology benchmark plan.')
    verify_plan_sources(plan,digest,source_root)
    if plan['execution']['paired_repetitions']!=7 or plan['execution']['memory_budget_bytes']!=2**31:
        raise ValueError('The reviewed repetition/memory policy must not be changed silently.')
    return plan,digest


def verify_plan_sources(plan,plan_sha256,source_root=ROOT):
    """Only the two main-approved public scripts may use their exact LF form."""
    binding_path=ROOT/'benchmarks/v2/learning-pathology-source-text-binding.json'
    if file_hash(binding_path)!=SOURCE_TEXT_BINDING_SHA256:
        raise ValueError('Source representation binding differs from the exact independently reviewed bytes.')
    binding=read_json(binding_path)
    whitelist={'tools/golden_workflow.py','tools/golden_models.py'}
    if binding['plan_sha256_unchanged']!=plan_sha256 or set(binding['files'])!=whitelist:
        raise ValueError('The source-text binding cannot authorize other plans or paths.')
    evidence={}
    for name,expected in plan['source_files_sha256'].items():
        raw=(Path(source_root)/name).read_bytes();actual=hashlib.sha256(raw).hexdigest()
        if name not in whitelist:
            if actual!=expected:raise ValueError('Frozen benchmark source/input differs: '+name)
            evidence[name]=dict(raw_sha256=actual,comparison='raw_bytes_exact')
            continue
        approved=binding['files'][name];canonical=raw.replace(b'\r\n',b'\n')
        if approved['original_plan_raw_sha256']!=expected or actual not in (expected,approved['accepted_LF_raw_sha256']):
            raise ValueError('Unapproved public-source byte representation: '+name)
        if b'\r' in canonical or raw.startswith(b'\xef\xbb\xbf') or hashlib.sha256(canonical).hexdigest()!=approved['canonical_LF_sha256']:
            raise ValueError('Canonicalization cannot accept a source/content edit: '+name)
        if len(canonical)!=approved['canonical_LF_bytes']:
            raise ValueError('Canonical source length changed: '+name)
        evidence[name]=dict(raw_sha256=actual,canonical_LF_sha256=approved['canonical_LF_sha256'],
            CRLF_count=raw.count(b'\r\n'),comparison='explicit_approved_CRLF_or_LF_only')
    return evidence


def pathology_model_plan(dataset):
    """Explicit patient sets from the predeclared eight-patient training formula."""
    records=dataset['records'];training=[row['patient_id'] for row in records if row['cohort']=='train']
    def index(patient):
        return int(patient.rsplit('P',1)[1])
    def part(population,validation):
        selected=set(validation)
        return dict(training_patients=sorted(set(population)-selected),validation_patients=sorted(selected))
    outer=[]
    for fold in range(2):
        split=part(training,[p for p in training if (index(p)//2)%2==fold])
        inner=[part(split['training_patients'],[p for p in split['training_patients'] if (index(p)//4)%2==inside]) for inside in range(2)]
        outer.append(dict(outer=split,inner=inner))
    final=[part(training,[p for p in training if (index(p)//2)%2==fold]) for fold in range(2)]
    classes=dataset['classes']
    return dict(schema_version=1,task='binary',seed=42,
        sample_patients={row['sample_id']:row['patient_id'] for row in records},
        sample_centers={row['sample_id']:row.get('center') for row in records},
        sample_classes={row['sample_id']:classes[row['target']] for row in records if row['cohort']=='train'},
        cohorts={cohort:sorted(row['patient_id'] for row in records if row['cohort']==cohort) for cohort in sorted({r['cohort'] for r in records})},
        classes=classes,patient_classes={row['patient_id']:[classes[row['target']]] for row in records if row['cohort']=='train'},
        outer_folds=outer,final_inner_folds=final,rng_scheme='explicit-patient-partitions-v1')


def prepare_inputs(plan,plan_sha256,output,source_root=ROOT):
    """Prepare byte inputs only. No QC/colors/aggregation/fitting/oracle timing."""
    import importlib.metadata
    import numpy as np
    from PIL import Image
    if np.__version__!=plan['environments']['common_versions']['numpy'] or importlib.metadata.version('Pillow')!=plan['environments']['common_versions']['Pillow']:
        raise RuntimeError('Deterministic input preparation requires the declared NumPy/Pillow versions.')
    output=Path(output)
    if output.exists():
        raise FileExistsError('Input preparation requires a fresh directory.')
    output.mkdir(parents=True)
    golden=Path(source_root)/'golden/python_golden_v1'
    model_inputs=read_json(golden/'model_inputs.json');plans=read_json(golden/'workflow_plans.json')
    learning=[]
    for declaration in plan['learning']['primary_cases']:
        task=declaration['task'];dataset=model_inputs['datasets'][task]
        path=output/('learning-'+task+'.json');write_json(path,dataset)
        split=output/('plan-'+task+'.json');write_json(split,plans[task])
        learning.append(dict(id=declaration['id'],task=task,dataset_path=path.name,
            dataset_sha256=file_hash(path),plan_path=split.name,plan_sha256=file_hash(split),
            rows=len(dataset['records']),patients=len({r['patient_id'] for r in dataset['records']})))
    clinical=read_json(golden/'statistics_reference.json')['clinical_logistic']['rows']
    write_json(output/'clinical-rows.json',clinical)
    generator=plan['pathology']['generator'];width,height=generator['width'],generator['height']
    images=output/'pathology-images';images.mkdir()
    rows=model_inputs['datasets']['binary']['records'][:generator['study_patients']*2]
    yy,xx=np.indices((height,width));pattern=((xx+2*yy)%17)[...,None]
    declarations=[];labels=[];seen=set()
    for row in rows:
        patient=row['patient_id'];slide=patient+'-S'+row['lesion_id']
        # Separate float64 arithmetic operations, as frozen by the plan.
        values=np.clip(np.asarray(row['values'][:3],dtype=np.float64),-2.,2.)
        base=np.asarray([120.+15.*values[0],40.+5.*values[1],160.+10.*values[2]])
        pixels=np.clip(base+pattern,0.,255.).astype(np.uint8)
        np.save(images/(slide+'.npy'),pixels,allow_pickle=False)
        Image.fromarray(pixels,mode='RGB').save(images/(slide+'.png'),compress_level=6,optimize=False)
        declarations.append(dict(patient_id=patient,slide_id=slide,source='benchmark-v2:pathology:'+patient+':'+slide,
            npy='pathology-images/'+slide+'.npy',png='pathology-images/'+slide+'.png',
            width=width,height=height,QC_sentinel=False))
        if patient not in seen:
            seen.add(patient);index=int(patient.rsplit('P',1)[1])
            labels.append(dict(sample_id=patient,patient_id=patient,center='synthetic',
                cohort='train' if index<8 else 'test' if index<10 else 'external_pathology',
                target=row['target'] if index<10 else None))
    patient=generator['extra_empty_patient'];slide=generator['extra_empty_slide']
    pixels=np.full((height,width,3),255,dtype=np.uint8)
    np.save(images/(slide+'.npy'),pixels,allow_pickle=False)
    Image.fromarray(pixels,mode='RGB').save(images/(slide+'.png'),compress_level=6,optimize=False)
    declarations.append(dict(patient_id=patient,slide_id=slide,source='benchmark-v2:pathology:'+patient+':'+slide,
        npy='pathology-images/'+slide+'.npy',png='pathology-images/'+slide+'.png',width=width,height=height,QC_sentinel=True))
    label_dataset=dict(classes=model_inputs['datasets']['binary']['classes'],records=labels)
    write_json(output/'pathology-images.json',declarations)
    write_json(output/'pathology-labels.json',label_dataset)
    write_json(output/'pathology-patient-plan.json',pathology_model_plan(label_dataset))
    files={str(path.relative_to(output)).replace('\\','/'):file_hash(path) for path in sorted(output.rglob('*')) if path.is_file()}
    manifest=dict(schema='radiomics.benchmark.v2.learning_pathology_inputs.v1',plan_sha256=plan_sha256,
        source_files_sha256=plan['source_files_sha256'],files_sha256=files,learning=learning,
        pathology=dict(images=len(declarations),study_patients=len(labels),total_patients=len({r['patient_id'] for r in declarations}),
            image_manifest='pathology-images.json',label_dataset='pathology-labels.json',patient_plan='pathology-patient-plan.json'),
        derived_patient_feature_input=dict(status='requires_approved_untimed_reference_preparation',
            scope='No patient feature matrix is fabricated by byte-input preparation. Generate it once with the frozen independent pathology spec only after plan approval.'))
    write_json(output/'manifest.json',manifest)
    return manifest


def case_declarations(plan,inputs,derived):
    base=[]
    for entry in inputs['learning']:
        base.append(dict(id=entry['id'],suite='machine_learning_statistics',level='complete_learning',
            dataset=entry['id'],operation=entry['id'],task=entry['task'],input_sha256=object_hash(entry),
            dataset_path=entry['dataset_path'],plan_path=entry['plan_path'],units=entry['patients'],unit_name='patients'))
    base.append(dict(id='clinical_observed_complete_case',suite='machine_learning_statistics',level='clinical_secondary',
        dataset='frozen_clinical_96rows',operation='clinical_observed_complete_case',input_sha256=inputs['files_sha256']['clinical-rows.json'],
        dataset_path='clinical-rows.json',units=96,unit_name='supplied_rows'))
    pixels_sha=object_hash({name:sha for name,sha in inputs['files_sha256'].items() if name.startswith('pathology-')})
    for name,kind in [('pathology_decoded_frontend','decoded'),('pathology_png_frontend','png')]:
        base.append(dict(id=name,suite='pathology',level='frontend',dataset='synthetic_12patients_600tiles',operation=name,
            input_kind=kind,input_sha256=pixels_sha,units=600,unit_name='tiles'))
    base.append(dict(id='pathology_patient_binary_model',suite='machine_learning_statistics',level='pathology_patient_model',
        dataset='reference_prepared_12patient_matrix',operation='pathology_patient_binary_model',task='binary',
        input_sha256=derived['dataset_sha256'],units=12,unit_name='patients',plan_path='pathology-patient-plan.json'))
    if plan['pathology_full_scene_diagnostic']['enabled']:
        base.append(dict(id='pathology_png_full_scene_diagnostic',suite='pathology',level='full_scene_diagnostic',
            dataset='synthetic_12patients_600tiles',operation='pathology_png_full_scene_diagnostic',input_kind='png',
            input_sha256=pixels_sha,units=600,unit_name='tiles'))
    result=[]
    for item in base:
        for state in plan['execution']['states']:
            case=dict(item,state=state,threads=1)
            case['configuration_sha256']=object_hash(dict(case=case,plan=plan))
            case['id']=item['id']+'-'+state;result.append(case)
    if len(base)!=8 or len(result)!=16 or len({row['id'] for row in result})!=16:
        raise ValueError('The complete predeclared 8×2 case grid must remain intact.')
    return result


def module_files(module):
    source=Path(module.__file__).resolve();root=source.parent
    single=None if hasattr(module,'__path__') else source.name
    files=[source] if single else sorted(root.rglob('*'))
    return dict(path=str(root),single_file=single,files_sha256={str(path.relative_to(root)).replace('\\','/'):file_hash(path)
        for path in files if path.is_file() and path.suffix.lower() in ('.py','.pyd','.so','.dll')})


def worker_setup(request):
    import importlib
    import importlib.metadata
    import importlib.util
    import numpy as np
    import pandas as pd
    import threadpoolctl
    plan=request['plan_contents'];probe=request.get('probe',False);backend=request['backend']
    modules={'numpy':np,'pandas':pd,'threadpoolctl':threadpoolctl}
    if backend=='rust':
        import ov_radiomics as rad
        from ov_radiomics import _native
        if 'site-packages' not in Path(rad.__file__).resolve().parts:
            raise RuntimeError('Native measurements require an installed immutable wheel, never an editable source tree.')
        if _native.IMPLEMENTATION_SHA256!=request['native_build']['core_implementation_sha256']:
            raise RuntimeError('Native core does not match the immutable source/build receipt.')
        if any(importlib.util.find_spec(name) is not None for name in ('sklearn','scipy','statsmodels','radiomics','SimpleITK','joblib','omicverse','wsidata')):
            raise RuntimeError('The dedicated native environment unexpectedly contains a prohibited reference backend.')
        modules['ov_radiomics']=rad
    else:
        if importlib.util.find_spec('ov_radiomics') is not None:
            raise RuntimeError('The independent reference environment must not contain the native project package.')
        level=request.get('case',{}).get('level')
        imports=[]
        if probe or level in ('complete_learning','pathology_patient_model','full_scene_diagnostic'):
            imports+=['sklearn.impute','sklearn.linear_model','sklearn.preprocessing','sklearn.metrics','joblib']
        if probe or level=='clinical_secondary':
            imports+=['scipy.stats','statsmodels.api']
        if probe or request.get('case',{}).get('input_kind')=='png':
            imports+=['PIL.Image']
        for name in imports:
            importlib.import_module(name)
        if probe:
            for name in ('sklearn','scipy','statsmodels','joblib','PIL'):
                modules[name]=importlib.import_module(name)
    if sys.version.split()[0]!='3.9.21' or np.__version__!='1.23.0' or pd.__version__!='1.3.5':
        raise RuntimeError('Both facade environments must retain the predeclared NumPy/pandas versions.')
    limiter=threadpoolctl.threadpool_limits(limits=1);pools=threadpoolctl.threadpool_info()
    recognized=[pool for pool in pools if pool.get('user_api') in ('blas','openmp')]
    if not recognized or any(pool.get('num_threads')!=1 for pool in recognized):
        raise RuntimeError('Matched single-thread numerical resources cannot be verified.')
    context=dict(plan=plan,backend=backend,limiter=limiter,pools=pools)
    if backend=='rust':
        context['rad']=modules['ov_radiomics']
    if probe:
        versions={name:importlib.metadata.version(name) for name in plan['environments']['common_versions']}
        if backend=='python':
            versions.update({name:importlib.metadata.version(name) for name in plan['environments']['reference_versions']})
        expected=dict(plan['environments']['common_versions'])
        if backend=='python':expected.update(plan['environments']['reference_versions'])
        if versions!=expected:
            raise RuntimeError('The reference/facade version set differs from the predeclared plan.')
        libraries={pool['filepath']:file_hash(pool['filepath']) for pool in pools if pool.get('filepath')}
        context['environment']=dict(executable=sys.executable,python_version=sys.version.split()[0],versions=versions,
            modules={name:module_files(module) for name,module in modules.items()},libraries_sha256=libraries,pools=pools)
        return context
    fixtures=Path(request['fixtures']);context['fixtures']=fixtures
    case=request['case']
    if case['level'] in ('frontend','full_scene_diagnostic') or request.get('derive_only'):
        from benchmark_lp_pathology import specification
        context['images']=read_json(fixtures/'pathology-images.json')
        context['pathology_config']=request['pathology_config']
        if backend=='python':
            source=Path(request['specification_root'])
            context['pixel_spec']=specification(source/'pathology_spec.py','benchmark_pixel_spec')
            context['aggregate_spec']=specification(source/'pathology_aggregate_spec.py','benchmark_aggregate_spec')
        if case.get('input_kind')=='decoded' or request.get('derive_only'):
            context['decoded']={row['npy']:np.load(fixtures/row['npy'],allow_pickle=False) for row in context['images']}
    if case['level']=='complete_learning':
        context['dataset']=read_json(fixtures/case['dataset_path']);context['split_plan']=read_json(fixtures/case['plan_path'])
    elif case['level']=='pathology_patient_model':
        context['dataset']=read_json(request['derived_dataset']);context['split_plan']=read_json(fixtures/case['plan_path'])
    elif case['level']=='clinical_secondary':
        context['rows']=read_json(fixtures/'clinical-rows.json')
    if case['level']=='full_scene_diagnostic':
        context['labels']=read_json(fixtures/'pathology-labels.json');context['split_plan']=read_json(fixtures/'pathology-patient-plan.json')
    return context


def workflow_call(context,case,output,stage):
    from benchmark_lp_learning import run_learning,run_clinical
    from benchmark_lp_pathology import run_frontend,patient_model_dataset
    backend=context['backend'];level=case['level']
    if level in ('complete_learning','pathology_patient_model'):
        return run_learning(context['dataset'],case['task'],context['split_plan'],backend,output,stage,
            fit_unit='patient' if level=='pathology_patient_model' else 'sample')
    if level=='clinical_secondary':
        return run_clinical(context['rows'],backend,output,stage)
    if level=='frontend':
        return run_frontend(context,case['input_kind'],backend,output,stage)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    front=run_frontend(context,'png',backend,output/'frontend',stage)
    with stage('model_fit_tune'):
        dataset=patient_model_dataset(front['hierarchy'],context['labels'],context['plan']['pathology_patient_model_case']['features'])
    fitted=run_learning(dataset,'binary',context['split_plan'],backend,output/'patient-model',stage,fit_unit='patient')
    return dict(comparison=lambda:dict(frontend=front['comparison'](),model=fitted['comparison']()))


def worker(request):
    import gc
    import time
    from benchmark_lp_runtime import Stages,lifetime_peak
    from benchmark_lp_learning import clean
    context=worker_setup(request)
    try:
        if request.get('probe'):
            return dict(status='ok',environment=context['environment'])
        case=request['case'];output=Path(request['observation_directory']);output.mkdir(parents=True,exist_ok=False)
        if request.get('derive_only'):
            from benchmark_lp_pathology import pixel_frontend,align_aggregate,patient_model_dataset
            stage=Stages(False);pixels=pixel_frontend(context,'decoded','python',stage)
            _,report=align_aggregate(context,pixels,'python',stage)
            dataset=patient_model_dataset(report,read_json(context['fixtures']/'pathology-labels.json'),
                context['plan']['pathology_patient_model_case']['features'])
            return dict(status='ok',derived_dataset=clean(dataset),validation_only=True)
        warmup=[]
        for index in range(request.get('warmup_calls',0)):
            started=time.perf_counter()
            result=workflow_call(context,case,output/f'warmup-{index}',Stages(False))
            seconds=time.perf_counter()-started
            warmup.append(dict(elapsed_seconds=seconds,comparison_values=result['comparison']()));del result
        gc.collect();measured=not request.get('validation_only',False);stage=Stages(measured)
        if measured:started=time.perf_counter()
        result=workflow_call(context,case,output/'timed' if measured else output/'validation',stage)
        elapsed=time.perf_counter()-started if measured else None
        artifact=result['comparison']()
        import threadpoolctl
        after=threadpoolctl.threadpool_info()
        if any(pool.get('num_threads')!=1 for pool in after if pool.get('user_api') in ('blas','openmp')):
            raise RuntimeError('A late-loaded numerical library escaped the fixed one-thread resource budget.')
        value=dict(status='ok',comparison_values=clean(artifact),warmup_values=warmup,
            peak_memory_bytes=lifetime_peak(),pools=context['pools'],pools_after=after,validation_only=not measured)
        if measured:
            value.update(elapsed_seconds=elapsed,stage_seconds=stage.seconds,
                unassigned_boundary_seconds=elapsed-sum(stage.seconds.values()))
        return value
    finally:
        context['limiter'].restore_original_limits()


def compare_values(expected,actual,plan,path='root'):
    """Predeclared layered comparisons, not finite-only or rounded digests."""
    import math
    failures=[]
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(expected)!=set(actual):
            return [dict(path=path,error='Exact object field set differs.')]
        for key in expected:
            failures+=compare_values(expected[key],actual[key],plan,path+'.'+str(key))
        return failures
    if isinstance(expected,list):
        if not isinstance(actual,list) or len(expected)!=len(actual):
            return [dict(path=path,error='Exact list length differs.')]
        for index,(a,b) in enumerate(zip(expected,actual)):
            failures+=compare_values(a,b,plan,path+f'[{index}]')
        return failures
    if expected is None or isinstance(expected,(str,bool)):
        return [] if type(expected) is type(actual) and expected==actual else [dict(path=path,reference=expected,actual=actual)]
    if not isinstance(actual,(int,float)) or isinstance(actual,bool) or not math.isfinite(float(actual)):
        return [dict(path=path,error='A finite numeric value became nonfinite or nonnumeric.')]
    if isinstance(expected,int):
        passed=actual==expected
    else:
        config=plan['correctness'];kind='color' if '.colors.' in path else 'prediction' if (
            '.probabilities[' in path or path.endswith('.value') or '.patient_reports[' in path or '.audit.' in path
            or any(token in path for token in ('.coefficient[','.standard_error[','.odds_ratio[','.ci_low[','.ci_high['))) else 'deterministic_scalar'
        passed=abs(actual-expected)<=config[kind+'_atol']+config[kind+'_rtol']*abs(expected)
    return [] if passed else [dict(path=path,reference=expected,actual=actual)]


def audit_request(request):
    load_plan(request['plan'],request['approved_plan_sha256'])
    for name,digest in request['runner_sources_sha256'].items():
        if file_hash(ROOT/name)!=digest:
            raise RuntimeError('Benchmark source changed after its reviewed binding: '+name)
    if request.get('probe'):
        return dict(status='verified_probe_control')
    environment=request['audit_environment']
    for name,module in environment['modules'].items():
        root=Path(module['path'])
        paths=[root/module['single_file']] if module.get('single_file') else sorted(root.rglob('*'))
        actual={str(path.relative_to(root)).replace('\\','/'):file_hash(path) for path in paths
            if path.is_file() and path.suffix.lower() in ('.py','.pyd','.so','.dll')}
        if actual!=module['files_sha256']:
            raise RuntimeError('A fixed numerical/source package changed: '+name)
    for path,digest in environment['libraries_sha256'].items():
        if file_hash(path)!=digest:
            raise RuntimeError('A fixed compiled numerical library changed.')
    for path,digest in request['input_files_sha256'].items():
        if file_hash(path)!=digest:
            raise RuntimeError('A predeclared input byte artifact changed: '+path)
    return dict(status='verified',environment_sha256=object_hash(environment),files=len(request['input_files_sha256']))


def finish_transport(result,request):
    if request.get('probe'):
        return result
    artifact=result.pop('comparison_values',None)
    directory=Path(request['observation_directory']);directory.mkdir(parents=True,exist_ok=True)
    if artifact is not None:
        write_json(directory/'comparison.json',artifact)
        result.update(comparison_artifact=str(directory/'comparison.json'),output_digest=object_hash(artifact))
    warmup=result.pop('warmup_values',[])
    for index,item in enumerate(warmup):
        value=item.pop('comparison_values')
        write_json(directory/f'warmup-{index}-comparison.json',value)
        item.update(output_digest=object_hash(value),comparison_artifact=str(directory/f'warmup-{index}-comparison.json'))
        if value!=artifact:
            result.update(status='failed',error='Warmup and measured outputs differ within one fixed worker.')
    result['warmup']=warmup
    files={str(path.relative_to(directory)).replace('\\','/'):dict(bytes=path.stat().st_size,sha256=file_hash(path))
        for path in sorted(directory.rglob('*')) if path.is_file()}
    result.update(output_files=files,output_bytes=sum(item['bytes'] for name,item in files.items() if name.startswith('timed/')))
    return result


RUNNER_FILES=['tools/benchmark_learning_pathology.py','tools/benchmark_lp_learning.py','tools/benchmark_lp_reference.py',
    'tools/benchmark_lp_pathology.py','tools/benchmark_lp_runtime.py','tools/benchmark_protocol.py']


def request_base(args,plan,plan_sha,backend,output):
    return dict(plan=str(Path(args.plan).resolve()),approved_plan_sha256=plan_sha,plan_contents=plan,backend=backend,
        source_text_binding_sha256=SOURCE_TEXT_BINDING_SHA256,source_text_evidence=verify_plan_sources(plan,plan_sha),
        runner_sources_sha256={name:file_hash(ROOT/name) for name in RUNNER_FILES},
        native_build=read_json(args.native_build_manifest),observation_directory=str(output),
        fixtures=str(Path(args.fixtures).resolve()),specification_root=str(ROOT/'golden/python_golden_v1'),
        pathology_config=read_json(ROOT/'golden/python_golden_v1/pathology_inputs.json'))


def launch_request(args,request,directory,*,measured):
    from benchmark_lp_runtime import launch
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    return launch(args.native_python if request['backend']=='rust' else args.reference_python,__file__,request,
        directory/'request.json',directory/'stdout.jsonl',directory/'stderr.txt',audit=audit_request,finalize=finish_transport,
        budget=request['plan_contents']['execution']['memory_budget_bytes'],timeout=request['plan_contents']['execution']['maximum_observation_seconds'],measured=measured)


def freeze_environments(args,plan,plan_sha):
    import zipfile
    from benchmark_lp_runtime import validate_receipt
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    receipt=read_json(args.native_build_manifest)
    validate_receipt(receipt,Path(args.native_wheel),Path(args.native_build_manifest).resolve().parent)
    binding=dict(schema='radiomics.benchmark.v2.learning_pathology_binding.v1',plan_sha256=plan_sha,native_build=receipt,
        source_text_binding_sha256=SOURCE_TEXT_BINDING_SHA256,source_text_evidence=verify_plan_sources(plan,plan_sha),
        native_build_manifest_sha256=file_hash(args.native_build_manifest),wheel_sha256=file_hash(args.native_wheel),
        runner_sources_sha256={name:file_hash(ROOT/name) for name in RUNNER_FILES},environments={})
    for backend in ('python','rust'):
        request=request_base(args,plan,plan_sha,backend,output/('probe-'+backend));request['probe']=True
        result=launch_request(args,request,output/('probe-transport-'+backend),measured=False)
        write_json(output/('probe-'+backend+'.json'),result)
        if result['status']!='ok':
            raise RuntimeError('Environment preflight failed; retain its probe artifact.')
        binding['environments'][backend]=result['environment']
    with zipfile.ZipFile(args.native_wheel) as archive:
        expected={name[len('ov_radiomics/'):]:hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
            if name.startswith('ov_radiomics/') and Path(name).suffix.lower() in ('.py','.pyd','.so','.dll')}
    if binding['environments']['rust']['modules']['ov_radiomics']['files_sha256']!=expected:
        raise RuntimeError('Installed project files do not exactly match the immutable wheel.')
    write_json(output/'binding.json',binding)
    return binding


def execution_request(args,plan,plan_sha,binding,case,output):
    request=request_base(args,plan,plan_sha,'python',output)
    fixtures=Path(args.fixtures).resolve();inputs=read_json(fixtures/'manifest.json')
    request.update(case=case,audit_environment=binding['environments']['python'],
        runner_sources_sha256=binding['runner_sources_sha256'],derived_dataset=args.derived_dataset,
        input_files_sha256={str(fixtures/name):digest for name,digest in inputs['files_sha256'].items()})
    if args.derived_dataset:
        request['input_files_sha256'][str(Path(args.derived_dataset).resolve())]=file_hash(args.derived_dataset)
    return request


def load_binding(args,plan_sha):
    if not args.binding or not args.approved_binding_sha256 or file_hash(args.binding)!=args.approved_binding_sha256:
        raise ValueError('An exact reviewed environment/artifact binding SHA is required.')
    binding=read_json(args.binding)
    if binding['plan_sha256']!=plan_sha:
        raise ValueError('Binding belongs to a different predeclared plan.')
    if binding['runner_sources_sha256']!={name:file_hash(ROOT/name) for name in RUNNER_FILES}:
        raise ValueError('Benchmark implementation changed after environment binding; freeze/review a new binding before measurement.')
    return binding


def freeze_derived(args,plan,plan_sha,binding):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    case=dict(id='derive-reference-patient-input',suite='pathology',level='frontend',input_kind='decoded',
        state='untimed_preparation',dataset='synthetic_12patients_600tiles',operation='derive_patient_input',
        threads=1,input_sha256=file_hash(Path(args.fixtures)/'manifest.json'),configuration_sha256=plan_sha)
    request=execution_request(args,plan,plan_sha,binding,case,output/'reference-derivation')
    request.update(derive_only=True,validation_only=True,warmup_calls=0)
    result=launch_request(args,request,output/'transport',measured=False)
    write_json(output/'derivation.json',result)
    if result['status']!='ok' or 'derived_dataset' not in result:
        raise RuntimeError('Untimed independent feature-input preparation failed; retain its evidence.')
    dataset=result.pop('derived_dataset');write_json(output/'patient-dataset.json',dataset)
    manifest=dict(schema='radiomics.benchmark.v2.pathology_patient_input.v1',plan_sha256=plan_sha,
        input_manifest_sha256=file_hash(Path(args.fixtures)/'manifest.json'),dataset_sha256=file_hash(output/'patient-dataset.json'),
        dataset_path=str((output/'patient-dataset.json').resolve()),reference_environment_sha256=object_hash(binding['environments']['python']),
        method='Untimed frozen independent Python pathology spec only; no native expected values or Golden files were recorded.')
    write_json(output/'manifest.json',manifest)
    return manifest


def retained_comparison(reference,native,plan):
    try:
        a=read_json(reference);b=read_json(native);failures=compare_values(a,b,plan)
        return dict(passed=not failures,mismatch_count=len(failures),first_mismatches=failures[:30])
    except Exception as exc:
        return dict(passed=False,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())


def validate_outputs(args,plan,plan_sha,binding,cases):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False);records=[]
    # Cold/warm share numerical definitions. Exactly one untimed run per base
    # workload is sufficient; no algorithm wall/API elapsed value is emitted.
    for case in cases[::2]:
        values={}
        for backend in ('python','rust'):
            request=execution_request(args,plan,plan_sha,binding,case,output/(case['operation']+'-'+backend))
            request.update(backend=backend,audit_environment=binding['environments'][backend],validation_only=True,warmup_calls=0)
            values[backend]=launch_request(args,request,output/(case['operation']+'-'+backend+'-transport'),measured=False)
        comparison=(retained_comparison(values['python'].get('comparison_artifact'),values['rust'].get('comparison_artifact'),plan)
            if all(row['status']=='ok' for row in values.values()) else dict(passed=False,error='At least one declared public workflow failed.'))
        records.append(dict(case=case,backends=values,comparison=comparison))
    write_json(output/'validation.json',dict(scope='Untimed correctness only; no API/wall speed observations',results=records))
    return records


def append_record(path,value):
    import os
    with Path(path).open('a',encoding='utf-8',newline='\n') as target:
        target.write(json.dumps(value,allow_nan=False,sort_keys=True)+'\n');target.flush();os.fsync(target.fileno())


def measure(args,plan,plan_sha,binding,cases):
    from benchmark_protocol import paired_summary,latency_summary,machine_manifest
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'plan.json',plan);write_json(output/'binding.json',binding)
    write_json(output/'provenance.json',dict(machine=machine_manifest(ROOT),runner_sources_sha256=binding['runner_sources_sha256']))
    summaries=[];raw=[]
    for index,case in enumerate(cases):
        directory=output/case['id'];directory.mkdir();records=[]
        for pair in range(7):
            attempts={}
            for order,backend in enumerate(('python','rust') if pair%2==0 else ('rust','python')):
                request=execution_request(args,plan,plan_sha,binding,case,directory/f'pair-{pair:02}-{backend}')
                request.update(backend=backend,audit_environment=binding['environments'][backend],
                    validation_only=False,warmup_calls=1 if case['state']=='warm_uncached' else 0)
                result=launch_request(args,request,directory/f'pair-{pair:02}-{backend}-transport',measured=True)
                result.update({key:case[key] for key in ('suite','level','dataset','operation','state','threads','input_sha256','configuration_sha256')})
                result.update(pair=pair,order=order,backend=backend,case_id=case['id'],binding_sha256=args.approved_binding_sha256,
                    source_text_binding_sha256=SOURCE_TEXT_BINDING_SHA256,correctness_checked=False)
                attempts[backend]=result
            comparison=(retained_comparison(attempts['python'].get('comparison_artifact'),attempts['rust'].get('comparison_artifact'),plan)
                if all(row['status']=='ok' for row in attempts.values()) else dict(passed=False,error='A backend execution/operation failed.'))
            for row in attempts.values():
                row.update(correctness_checked=comparison['passed'],comparison=comparison)
                records.append(row);raw.append(row);append_record(output/'raw-observations.jsonl',row)
        summary=paired_summary(records,repeats=7);summary['case']=case
        if summary['valid_speed_comparison']:
            memory={backend:latency_summary([row['peak_memory_bytes'] for row in records if row['backend']==backend]) for backend in ('python','rust')}
            memory_ratio=memory['rust']['median_seconds']/memory['python']['median_seconds']
            summary.update(runtime_native_over_reference=1/summary['ratio_python_over_rust'],
                runtime_target_miss=1/summary['ratio_python_over_rust']>1.10,memory_target_miss=memory_ratio>1.20,
                peak_memory_comparison=memory_ratio,memory_status='Full process lifetime resident memory; bytes, not traced allocations.',
                peak_memory={backend:{key.replace('_seconds','_bytes'):value for key,value in detail.items() if key!='p95_interpretation'} for backend,detail in memory.items()},
                throughput={backend:case['units']/summary['backends'][backend]['median_seconds'] for backend in ('python','rust')},
                throughput_unit=case['unit_name']+'/s')
            stages={}
            for backend in ('python','rust'):
                names=sorted({key for row in records if row['backend']==backend for key in row.get('stage_seconds',{})})
                stages[backend]={name:latency_summary([row['stage_seconds'][name] for row in records if row['backend']==backend]) for name in names}
            summary['stage_summaries']=stages
        summaries.append(summary);write_json(directory/'summary.json',summary)
        print(f"[{index+1}/{len(cases)}] {case['id']}: "+str(summary.get('ratio_python_over_rust','invalid')),flush=True)
    by_suite={}
    for suite in plan['suites']:
        selected=[row for row in summaries if row['case']['suite']==suite]
        by_suite[suite]=dict(results=selected,correctness_status='passed_recorded_cases' if all(row['valid_speed_comparison'] for row in selected) else 'failed_or_incomplete_cases',
            policy='No pooled speedup or cross-suite total; misses remain individual rows.')
    complete=len(summaries)==16 and len(raw)==224
    report=dict(schema='radiomics.benchmark.v2.learning_pathology_report.v1',plan_sha256=plan_sha,
        source_text_binding_sha256=SOURCE_TEXT_BINDING_SHA256,source_text_evidence=verify_plan_sources(plan,plan_sha),
        binding_sha256=args.approved_binding_sha256,complete_declared_grid=complete,expected_cases=16,expected_observations=224,
        suites=by_suite,raw_observations='raw-observations.jsonl',performance_targets=dict(runtime_ratio=1.10,memory_ratio=1.20),
        missing_evidence=['No multicore scaling','No bootstrap/DeLong predictive-CI runtime','OS disk cache/background load uncontrolled',
            'Independent final artifact installation and Python-free CLI evidence remain separately scoped'])
    write_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default=str(ROOT/'benchmarks/v2/learning-pathology-plan.json'))
    parser.add_argument('--approved-plan-sha256')
    parser.add_argument('--prepare-inputs',action='store_true')
    parser.add_argument('--freeze-environments',action='store_true')
    parser.add_argument('--freeze-derived',action='store_true')
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--fixtures')
    parser.add_argument('--native-python')
    parser.add_argument('--reference-python')
    parser.add_argument('--native-wheel')
    parser.add_argument('--native-build-manifest')
    parser.add_argument('--binding')
    parser.add_argument('--approved-binding-sha256')
    parser.add_argument('--derived-dataset')
    parser.add_argument('--derived-manifest')
    parser.add_argument('--worker',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        from benchmark_lp_runtime import emit_worker,lifetime_peak
        from benchmark_lp_learning import clean
        try:
            result=worker(read_json(args.worker))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            result=dict(status='failed',error=f'{type(exc).__name__}: {exc}',peak_memory_bytes=lifetime_peak())
        emit_worker(clean(result))
        return 0 if result['status']=='ok' else 1
    if sum((args.prepare_inputs,args.freeze_environments,args.freeze_derived,args.validate,args.run))!=1 or not args.output:
        parser.error('Choose one explicit operation and a fresh --output directory.')
    if not args.prepare_inputs and not all((args.approved_plan_sha256,args.fixtures,args.native_python,args.reference_python,args.native_wheel,args.native_build_manifest)):
        parser.error('Execution requires the reviewed plan SHA, inputs, dedicated interpreters and immutable wheel/build receipt.')
    plan,digest=load_plan(args.plan,args.approved_plan_sha256)
    if args.prepare_inputs:
        prepare_inputs(plan,digest,args.output)
        print('Prepared deterministic bytes only; no algorithms were measured or Golden outputs regenerated.')
        return 0
    if args.freeze_environments:
        freeze_environments(args,plan,digest)
        print('Froze artifact/environment metadata only; no algorithms were measured.')
        return 0
    binding=load_binding(args,digest)
    if args.freeze_derived:
        freeze_derived(args,plan,digest,binding)
        print('Prepared the independent pathology patient input without timing or Golden changes.')
        return 0
    if not args.derived_dataset or not args.derived_manifest:
        parser.error('Validation/measurement requires the fixed independent patient-feature dataset and its manifest.')
    inputs=read_json(Path(args.fixtures)/'manifest.json');derived=read_json(args.derived_manifest)
    if inputs['plan_sha256']!=digest or derived['plan_sha256']!=digest or derived['dataset_sha256']!=file_hash(args.derived_dataset):
        raise ValueError('Input/derived artifacts differ from the reviewed immutable configuration.')
    declared=case_declarations(plan,inputs,derived)
    if args.validate:
        records=validate_outputs(args,plan,digest,binding,declared)
        return 0 if all(record['comparison']['passed'] for record in records) else 1
    report=measure(args,plan,digest,binding,declared)
    return 0 if report['complete_declared_grid'] and all(suite['correctness_status']=='passed_recorded_cases'
        for suite in report['suites'].values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
