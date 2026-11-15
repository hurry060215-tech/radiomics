"""Python data views of native patient-grouped learning. No Python estimators."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ._native_common import request, study, string_value, prediction_frame, tabular
from ._native_selection import NativeFeatureSelector
from ._types import ModelBundle, PredictionTable
from .split import SplitPlan


def _model_spec(name, task, params, seed):
    defaults = {
        'LR':dict(model='LR', **{'lambda':.1},tolerance=1e-10,max_iterations=5000),
        'Linear':dict(model='Linear',rank_tolerance=1e-12),
        'Ridge':dict(model='Ridge',**{'lambda':.1},rank_tolerance=1e-12),
        'KNN':dict(model='KNN',k=5,weights='uniform'),
        'GaussianNB':dict(model='GaussianNB',priors=None,var_smoothing=1e-9),
        'RandomForest':dict(model='RandomForest',trees=200,max_depth=None,min_samples_split=2,
            min_samples_leaf=1,max_features=dict(method='all' if task=='regression' else 'sqrt'),bootstrap=True,seed=seed),
    }
    if name not in defaults:
        raise ValueError(f'{name} has no supported native specification; select backend=python explicitly for an external model.')
    result = defaults[name]
    aliases = {'n_estimators':'trees','n_neighbors':'k','max_iter':'max_iterations','tol':'tolerance','random_state':'seed'}
    for raw,value in params.items():
        key = aliases.get(raw,raw)
        if key not in result or key == 'model':
            raise ValueError(f'Unsupported native {name} parameter: {raw}; native regularization uses explicit mean-loss lambda.')
        if name == 'RandomForest' and key == 'max_features' and not isinstance(value,dict):
            value = dict(method='all') if value is None or value=='all' else dict(method='sqrt') if value=='sqrt' else dict(method='count',count=value)
        result[key] = value
    return result


class NativeTrainingResult:
    def __init__(self, payload, study_data=None, api_config=None):
        self._payload = payload
        self._study = study_data
        self.config = deepcopy(api_config or {})
        self.features = list(self.config.get('features',payload['models'][0]['model']['manifest']['feature_names']))
        self.task = payload['config']['task']
        self.classes = payload['config']['classes']
        self.split_plan = SplitPlan.from_dict(payload['split_plan'])
        self.best_model = payload['best_model']
        self.models = {run['name']:ModelBundle(run['model']) for run in payload['models']}
        truth = None
        if study_data is not None:
            outcome = study_data['outcome']
            if outcome['task'] != 'unlabeled':
                truth = {s['sample_id']:v for s,v in zip(study_data['samples'],outcome['values'])}
        self.oof_predictions = pd.concat([prediction_frame(run['oof'],run['name'],truth) for run in payload['models']],ignore_index=True)
        self.predictions = pd.concat([prediction_frame(run['predictions'],run['name'],truth) for run in payload['models']],ignore_index=True)
        metric = {'binary':'auc','multiclass':'macro_ovr_auc','regression':'rmse'}[self.task]
        self.cv_results = pd.DataFrame([dict(model=run['name'],fold=f['fold'],score=f['score'],
            **{metric:f['score']}) for run in payload['models'] for f in run['outer_folds']])

    def to_dict(self):
        return deepcopy(self._payload)

    def predict_proba(self,data,model='LR'):
        probabilities = self.models[model].predict_proba(data)
        return (pd.Series(probabilities[:,1],index=data.index,name=model) if self.task=='binary'
            else pd.DataFrame(probabilities,index=data.index,columns=self.classes))

    def predict(self,data,model=None,*,cohort='external'):
        model = model or self.best_model
        bundle = self.models[model]
        fitted = bundle.to_dict()['fitted']['preprocessing']
        numeric = [s['name'] for s in fitted['numeric']]
        categorical = [s['name'] for s in fitted['categorical']]
        converted,_ = study(data,None,numeric+categorical,task=self.task,classes=self.classes,
            categorical=categorical,split=self.config.get('split','group'),
            id_col=self.config.get('id_col','ID'),patient_id=self.config.get('patient_id'),default_cohort=cohort)
        return bundle.predict_study(converted)

    def save(self,directory,*,overwrite=False):
        path = Path(directory)
        known = ['training.json','split_plan.json','api_config.json','cv_results.csv','oof_predictions.csv','predictions.csv']
        known += [f'model-{i:03}.rad' for i in range(len(self.models))]
        if not overwrite and path.exists() and any((path/name).exists() for name in known):
            raise FileExistsError('Native output exists; choose a new directory or pass overwrite=True.')
        path.mkdir(parents=True,exist_ok=True)
        for i,bundle in enumerate(self.models.values()):
            bundle.save(path/f'model-{i:03}.rad',overwrite=overwrite)
        for name,value in [('training.json',self._payload),('split_plan.json',self.split_plan.to_dict()),('api_config.json',self.config)]:
            (path/name).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        for name in ('cv_results','oof_predictions','predictions'):
            getattr(self,name).to_csv(path/(name+'.csv'),index=False)


def train_models(data,target,features,split='group',models=None,selector=None,
                 continuous=None,categorical=None,scale=None,cv=5,scoring=None,
                 model_params=None,random_state=0,id_col='ID',positive_label=None,
                 patient_id=None,smote=False,*,task='binary',split_plan=None,
                 inner_cv=3,n_jobs=1,classes=None,fit_unit='sample',scoring_unit='patient',
                 threshold=.5,native_config=None):
    if smote:
        raise ValueError('SMOTE is an explicit optional Python backend, not part of native training.')
    if task not in ('binary','multiclass','regression'):
        raise ValueError('task must be binary, multiclass, or regression.')
    expected = {'binary':('roc_auc',),'multiclass':('macro_ovr_auc','roc_auc_ovr'),'regression':('rmse','neg_root_mean_squared_error')}[task]
    if scoring is not None and scoring not in expected:
        raise ValueError('Native model comparison uses the declared task metric, not an implicit alternative scorer.')
    if split_plan is not None:
        plan = split_plan if isinstance(split_plan,SplitPlan) else SplitPlan.from_dict(split_plan)
        classes = plan.to_dict()['classes']
    else:
        if split not in data:
            raise ValueError('Native training requires explicit patient-level cohort assignments.')
        training = data.loc[data[split]=='train',target]
        if training.isna().any():
            raise ValueError('Training outcomes must be observed.')
        if task == 'regression':
            classes = []
        elif classes is None:
            classes = sorted({string_value(v) for v in training})
            if task == 'binary':
                positive = string_value(1 if positive_label is None else positive_label)
                if len(classes)!=2 or positive not in classes:
                    raise ValueError('Specify the binary positive_label present in training outcomes.')
                classes = [c for c in classes if c!=positive]+[positive]
        plan = None
    classes=[string_value(value) for value in classes]
    if task=='binary':
        if len(classes)!=2:
            raise ValueError('Binary training requires exactly two ordered classes.')
        if positive_label is not None and string_value(positive_label)!=classes[1]:
            raise ValueError('positive_label must agree with the second entry of the explicit class order/SplitPlan.')
        positive_label=classes[1]
    elif task=='multiclass' and len(classes)<3:
        raise ValueError('Multiclass training requires at least three ordered classes.')
    converted,group = study(data,target,list(features),task=task,classes=classes,
        categorical=categorical,split=split,id_col=id_col,patient_id=patient_id)
    if continuous is not None and (set(continuous)&set(categorical or []) or set(continuous)|set(categorical or [])!=set(features)):
        raise ValueError('Continuous and categorical names must partition the features.')
    if plan is None:
        plan = SplitPlan.from_data(data,target,task=task,classes=classes,id_col=id_col,
            patient_id=group,split=split,center='center' if 'center' in data else None,
            outer_cv=cv,inner_cv=inner_cv,seed=random_state)
    if native_config is None:
        preprocessing = dict(scaling=scale or 'none',all_missing='error')
        selections = [None]
        if selector is not None:
            if not isinstance(selector,NativeFeatureSelector):
                raise ValueError('A native pipeline requires a native FeatureSelector configuration; no Python selector is executed.')
            preprocessing,selections = selector.configurations(task)
        names = list(models) if models is not None else ['Ridge' if task=='regression' else 'LR']
        params = model_params or {}
        if set(params)-set(names):
            raise ValueError('model_params contains a model not requested.')
        grids=[]
        for name in names:
            choices=params.get(name,{})
            choices=choices if isinstance(choices,list) else [choices]
            candidates=[dict(preprocessing=preprocessing,selector=selection,
                model=_model_spec(name,task,choice,random_state))
                for choice,selection in itertools.product(choices,selections)]
            grids.append(dict(name=name,candidates=candidates))
        native_config=dict(task=task,classes=list(classes),fit_unit=fit_unit,scoring_unit=scoring_unit,
            threshold=threshold if task=='binary' else None,selection_tie_atol=1e-12,models=grids)
    else:
        native_config=deepcopy(native_config)
    payload=request('train',study=converted,plan=plan.to_dict(),config=native_config,n_jobs=n_jobs)
    return NativeTrainingResult(payload,converted,dict(target=target,split=split,id_col=id_col,
        patient_id=group,features=list(features),categorical=list(categorical or []),task=task,backend='rust',
        sparse_grid_metric='whole_pipeline_task_metric',positive_label=string_value(positive_label)))


class NativeFusionData:
    """Immutable view of probabilities plus complete native training provenance."""
    def __init__(self,results,model='LR',cohort='oof'):
        if not results or not all(isinstance(r,NativeTrainingResult) for r in results.values()):
            raise TypeError('Supply named native TrainingResults, not unverified fold-number tables.')
        self.results=dict(results);self.model_name=model;self.cohort=cohort
        first=next(iter(results.values()))
        for result in results.values():
            first.split_plan.assert_shared(result.split_plan)
        self.raw_tables={};frames=[]
        self.classes=first.classes;self.task=first.task
        for name,result in results.items():
            run=next((r for r in result._payload['models'] if r['name']==model),None)
            if run is None:
                raise ValueError('Requested fusion base model is absent: '+model)
            table=deepcopy(run['oof'] if cohort=='oof' else run['predictions'])
            if cohort!='oof':
                table['rows']=[r for r in table['rows'] if r['cohort']==cohort]
            if not table['rows']:
                raise ValueError('No predictions for fusion cohort: '+cohort)
            self.raw_tables[name]=table
            grouped=request('aggregate_patient_predictions',predictions=table)
            frame=pd.DataFrame(dict(ID=[r['patient_id'] for r in grouped['rows']],
                group=[r['cohort'] for r in grouped['rows']],fold=[r['fold'] for r in grouped['rows']]))
            for k,label in enumerate(self.classes):
                frame[name if self.task=='binary' and k==1 else name+'::'+label]=[r['probabilities'][k] for r in grouped['rows']]
            if frames:
                previous=frames[0].set_index('ID')
                current=frame.set_index('ID')
                if set(previous.index)!=set(current.index) or not previous[['group','fold']].sort_index().equals(current[['group','fold']].sort_index()):
                    raise ValueError('Fusion patient/cohort/fold identities differ.')
            frames.append(frame)
        merged=frames[0]
        for frame in frames[1:]:
            merged=merged.merge(frame.drop(columns=['group','fold']),on='ID',validate='one_to_one')
        truth=_truth_groups(first._study,'patient',selected_ids=set(merged.ID)) if first._study is not None else {}
        if any(value[0] is not None and value[0] not in self.classes for value in truth.values()):
            raise ValueError('The requested fusion view contains an observed unknown training class.')
        merged['y_true']=[self.classes.index(truth[i][0]) if i in truth and truth[i][0] in self.classes else np.nan for i in merged.ID]
        merged['prediction_type']='oof' if cohort=='oof' else 'apparent' if cohort=='train' else 'holdout' if cohort=='test' else 'external'
        self._frame=merged

    @property
    def frame(self):
        return self._frame.copy(deep=True)

    def __getitem__(self,key):
        return self._frame[key].copy(deep=True)

    def __getattr__(self,name):
        if name in self._frame.columns:
            return self._frame[name].copy(deep=True)
        raise AttributeError(name)

    def __len__(self):
        return len(self._frame)


class NativeFusionResult:
    def __init__(self,payload,inputs=None):
        self._payload=payload;self.model=ModelBundle(payload['model'])
        self.score_columns=list(payload['model']['fitted']['meta_feature_names'])
        self._study=None if not inputs else next(iter(inputs.values()))._study
        self.predictions=prediction_frame(payload['predictions'],'Fusion')

    def predict_proba(self,data):
        if not isinstance(data,NativeFusionData) or data.cohort=='oof':
            raise ValueError('Predict fusion from verified final base predictions; base OOF scores are meta training, not final prediction inputs.')
        table=request('predict_fusion_tables',model=self.model.to_dict(),tables=data.raw_tables)
        frame=prediction_frame(table,'Fusion').set_index('ID')
        if table['task']=='binary':
            return pd.Series(frame.loc[data._frame.ID,'probability'].to_numpy(),index=data._frame.index,name='Fusion')
        return frame.loc[data._frame.ID,['probability::'+c for c in table['classes']]].set_axis(table['classes'],axis=1)

    def save(self,path,*,overwrite=False):
        return self.model.save(path,overwrite=overwrite)


def fusion_data(results,model='LR',cohort='oof'):
    return NativeFusionData(results,model,cohort)


def fit_fusion(inputs,target='y_true',score_columns=('Rad','Clinical'),model='LR',random_state=0,
               *,method='logistic',weights=None,threshold=.5,lambda_=0.1):
    if model!='LR':
        raise ValueError('Native probability meta modeling uses Logistic/softmax LR.')
    if isinstance(inputs,NativeFusionData):
        if inputs.cohort!='oof':
            raise ValueError('Fusion fitting requires base OOF training predictions.')
        sources=inputs.results;base_name=inputs.model_name
    elif isinstance(inputs,dict):
        sources=inputs;base_name=model
    else:
        raise TypeError('Native fusion requires verified native results/shared SplitPlan, not a plain score DataFrame.')
    if not sources or not all(isinstance(r,NativeTrainingResult) for r in sources.values()):
        raise TypeError('Every fusion source must be a native TrainingResult.')
    if method=='weighted':
        if weights is None:
            raise ValueError('Weighted fusion requires explicit component weights.')
        if isinstance(weights,dict):
            if set(weights)!=set(sources):
                raise ValueError('Weight names must match fusion components exactly.')
            weights=[weights[name] for name in sources]
        specification=dict(method='weighted',weights=list(weights))
    elif method=='logistic':
        specification=dict(method='logistic',**{'lambda':lambda_},tolerance=1e-10,max_iterations=5000)
    else:
        raise ValueError('Fusion method must be logistic or weighted.')
    first=next(iter(sources.values()))
    payload=request('fit_fusion',inputs=[dict(name=name,training=result.to_dict(),model_name=base_name)
        for name,result in sources.items()],config=dict(method=specification,threshold=threshold if first.task=='binary' else None))
    return NativeFusionResult(payload,sources)


def _truth_groups(converted,unit,selected_ids=None):
    outcome=converted['outcome'];values=outcome.get('values',[None]*len(converted['samples']))
    groups={}
    for sample,cohort,value in zip(converted['samples'],converted['cohorts'],values):
        key=sample['patient_id'] if unit=='patient' else sample['sample_id']
        if selected_ids is not None and key not in selected_ids:
            continue
        if key in groups and (groups[key][0]!=value or groups[key][1]!=cohort or groups[key][2]!=sample['patient_id']):
            raise ValueError('Evaluation/fusion requires consistent targets, cohorts and patient identities at the declared unit.')
        groups[key]=(value,cohort,sample['patient_id'])
    return groups


def _center_tables(source,converted):
    unit=source['unit'];centers={}
    for sample in converted['samples']:
        key=(sample['patient_id'] if unit=='patient' else sample['slide_id'] if unit=='slide' else sample['sample_id'])
        centers.setdefault(key,set()).add(sample.get('center'))
    partitions={}
    for row in source['rows']:
        key=row['patient_id'] if unit=='patient' else row['slide_id'] if unit=='slide' else row['sample_id']
        values=centers.get(key,set())
        if len(values)!=1 or next(iter(values)) is None or not next(iter(values)).strip():
            raise ValueError('Center evaluation needs one known center per actual prediction unit; an already aggregated multi-center score cannot be disaggregated.')
        center=next(iter(values));partitions.setdefault(center,[]).append(row)
    return [(center,dict(source,rows=rows)) for center,rows in sorted(partitions.items())]


def evaluate(predictions,threshold=None,ci_method='auto',confidence=.95,*,prediction_unit='patient',by_center=False,
             n_bootstrap=2000,random_state=0,data=None,target=None,task=None,classes=None,
             id_col='ID',patient_id=None,split='group'):
    from .metrics import _native_binary_report
    if isinstance(predictions,NativeTrainingResult):
        tables=[(run['name'],run['predictions']) for run in predictions._payload['models']]
        converted=predictions._study;task=predictions.task;classes=predictions.classes
        thresholds={run['name']:run['model']['manifest']['threshold'] for run in predictions._payload['models']}
    elif isinstance(predictions,NativeFusionResult):
        tables=[('Fusion',predictions._payload['predictions'])]
        converted=predictions._study;task=tables[0][1]['task'];classes=tables[0][1]['classes']
        thresholds={'Fusion':predictions.model.manifest['threshold']}
    elif isinstance(predictions,PredictionTable):
        tables=[('Model',predictions.to_dict())];task=predictions.to_dict()['task'];classes=predictions.to_dict()['classes']
        if task=='binary' and threshold is None:
            raise ValueError('A standalone binary PredictionTable requires an explicit saved-model threshold.')
        thresholds={'Model':threshold}
        if data is None:
            raise ValueError('Native prediction-table evaluation requires explicitly identified truth data.')
        converted,_=study(data,target,[],task=task,classes=classes,split=split,id_col=id_col,patient_id=patient_id)
    else:
        raise TypeError('Native evaluate expects a native result or PredictionTable with truth data.')
    if converted is None:
        raise ValueError('No identified outcomes are available; provide truth data to evaluate saved predictions.')
    if prediction_unit not in ('sample','patient'):
        raise ValueError('Choose explicit sample or patient evaluation.')
    truth=_truth_groups(converted,prediction_unit)
    rows=[];details=[]
    tables=[(name,center,source) for name,table in tables for center,source in
        (_center_tables(table,converted) if by_center else [(None,table)])]
    for name,center,source in tables:
        effective_threshold=threshold if threshold is not None else thresholds[name]
        effective_threshold=.5 if effective_threshold is None else effective_threshold
        if prediction_unit=='sample' and source['unit']!='sample':
            raise ValueError('Patient/slide/tile predictions cannot be broadcast as independent sample predictions.')
        table=request('aggregate_patient_predictions',predictions=source) if prediction_unit=='patient' and source['unit']!='patient' else source
        for cohort in sorted({r['cohort'] for r in table['rows']}):
            subset=[r for r in table['rows'] if r['cohort']==cohort];outcomes=[]
            for row in subset:
                key=row['patient_id'] if prediction_unit=='patient' else row['sample_id']
                if key not in truth or truth[key][1:]!=(row['cohort'],row['patient_id']):
                    raise ValueError('Prediction/truth identity or cohort differs.')
                outcomes.append(truth[key][0])
            result=dict(model=name,group=cohort,task=task,prediction_unit=prediction_unit,n=len(subset))
            if by_center:
                result['center']=center
            subset_table=dict(table,rows=subset)
            kinds=sorted(set(prediction_frame(subset_table).prediction_type))
            result['prediction_type']=','.join(kinds)
            if all(v is None for v in outcomes):
                result['status']='prediction_only';rows.append(result);continue
            if any(v is None for v in outcomes):
                raise ValueError('Partly missing evaluation outcomes cannot be silently discarded.')
            method=('delong' if task=='binary' else 'bootstrap') if ci_method=='auto' else ci_method
            if task=='regression':
                prediction=dict(task='regression',values=[r['value'] for r in subset]);target_value=dict(task='regression',values=outcomes)
                report=request('regression_metrics',y=outcomes,prediction=prediction['values'])
            else:
                if any(v not in classes for v in outcomes):
                    raise ValueError('An observed evaluation outcome uses an unknown training class.')
                y=[classes.index(v) for v in outcomes]
                prediction=dict(task='classification',classes=classes,probabilities=dict(rows=len(subset),cols=len(classes),
                    values=[p for r in subset for p in r['probabilities']],missing=[False]*(len(subset)*len(classes))))
                target_value=dict(task='classification',classes=classes,labels=y)
                if task=='binary':
                    report=_native_binary_report(np.asarray(y),np.asarray([r['probabilities'][1] for r in subset]),
                        effective_threshold,method,confidence,n_bootstrap,random_state)
                else:
                    report=request('classification_metrics',task=task,target=target_value,prediction=prediction,threshold=effective_threshold)
            result.update({k:float(v) if isinstance(v,str) and v in ('NaN','Inf','-Inf') else v for k,v in report.items() if not isinstance(v,(dict,list))})
            for average in ('macro','micro','weighted'):
                for key,value in report.get(average+'_average',{}).items():
                    result[average+'_'+key]=value
            if task!='binary' and method is not None:
                if method!='bootstrap':
                    raise ValueError('DeLong/H-L/DCA overall semantics are binary only; use native bootstrap for multiclass/regression.')
                interval=request('bootstrap_'+task,target=target_value,prediction=prediction,
                    replicates=n_bootstrap,seed=random_state,confidence=confidence)
                result.update(ci_low=interval['low'],ci_high=interval['high'])
            result['status']='evaluated';rows.append(result)
            details.append(dict(model=name,group=cohort,center=center,prediction_unit=prediction_unit,report=report))
    output=pd.DataFrame(rows)
    output.attrs['native_reports']=details
    return output
