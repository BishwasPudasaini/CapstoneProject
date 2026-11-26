from pathlib import Path
import argparse,json,platform
import numpy as np
from engine import ROOT,FEATURE_VERSION,features,video_frames,procedure,signature,digest

def train(manifest_path,output):
    import sklearn
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score,confusion_matrix
    manifest_path=Path(manifest_path).resolve(); data=json.loads(manifest_path.read_text(encoding='utf-8'))
    p=procedure(); classes={s['id'] for s in p['steps']}
    if data.get('include_unknown'): classes.add('unknown')
    split_rows={s:[] for s in ['train','validation','test']}; groups={}; hashes=set(); ids=set()
    synthetic=data.get('synthetic') is True
    window=data.get('smoothing_frames',3)
    if not isinstance(window,int) or not 1<=window<=10: raise ValueError('smoothing_frames must be an integer from 1 to 10.')
    training_only=data.get('training_only') is True
    if training_only and any(i['split']!='train' for i in data['sessions']):
        raise ValueError('Training-only mode accepts only training recordings.')
    for item in data['sessions']:
        split=item['split']; sid=item['session_id']; group=item['source_group']
        if split not in split_rows: raise ValueError('split must be train, validation or test.')
        if sid in ids: raise ValueError('Duplicate session_id.')
        ids.add(sid)
        if group in groups and groups[group]!=split: raise ValueError('Source group appears in multiple splits.')
        groups[group]=split
        if not synthetic and item.get('permission_confirmed') is not True:
            raise ValueError('Confirm permission for each real recording.')
        video=(manifest_path.parent/item['video']).resolve()
        if not video.is_relative_to(manifest_path.parent): raise ValueError('Video must be inside the manifest folder.')
        if not video.is_file(): raise ValueError(f'Missing video: {video}')
        h=digest(video)
        if h in hashes: raise ValueError('Duplicate video bytes. Use independent recordings.')
        hashes.add(h)
        segments=item['segments']; end=-1
        for seg in segments:
            a,b=seg['start'],seg['end']
            if not isinstance(a,(int,float)) or not isinstance(b,(int,float)) or not np.isfinite([a,b]).all() or not 0<=a<b or a<end or seg['step_id'] not in classes:
                raise ValueError(f'{sid}: invalid interval {a}–{b} for {seg.get("step_id")}. Enter end > start, with no overlap and a valid step label.')
            end=b
        samples=[]; last_time=0.
        for t,frame in video_frames(video):
            last_time=t
            label=next((s['step_id'] for s in segments if s['start']<=t<s['end']),None)
            if label is not None: samples.append((features(frame),label,t))
        if end>last_time+.6: raise ValueError('Label interval extends past video duration (allowance 0.6 sec).')
        if not samples: raise ValueError(f'No labelled frames in {sid}.')
        split_rows[split].append((sid,samples))
    for split,rows in split_rows.items():
        if training_only and split!='train': continue
        found={x[1] for _,samples in rows for x in samples}
        if found!=classes: raise ValueError(f'{split} must include every step class. Found: {sorted(found)}')
    if len({i['source_group'] for i in data['sessions'] if i['split']=='train'})<2:
        raise ValueError('Use at least two independent training source groups.')
    def flatten(split):
        rows=[x for _,samples in split_rows[split] for x in samples]
        return np.stack([x[0] for x in rows]),np.asarray([x[1] for x in rows])
    X,y=flatten('train')
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42))
    model.fit(X,y)
    report={'dataset':'USER-LABELLED GENERATED FOOTAGE' if synthetic else 'USER-LABELLED REAL LEGO',
            'train_samples':len(y),'classes':model.classes_.tolist(),'metrics':{},
            'note':'Test data is held out. Do not tune using test results. Scores do not measure physical build correctness.'}
    report['evaluation_status']='not_evaluated' if training_only else 'held_out_evaluation'
    if training_only: report['note']='Trained on two or more source recordings.'
    for split in ([] if training_only else ['validation','test']):
        Xt,yt=flatten(split);pred=model.predict(Xt)
        smooth_pred=[]
        for _,samples in split_rows[split]:
            probs=model.predict_proba(np.stack([x[0] for x in samples]))
            for i in range(len(probs)):
                # Avoid averaging across unlabelled gaps.
                first=i
                while first>max(0,i-window+1) and samples[first][2]-samples[first-1][2]<=1.5: first-=1
                smooth_pred.append(model.classes_[probs[first:i+1].mean(axis=0).argmax()])
        report['metrics'][split]={'samples':len(yt),'frame_macro_f1':float(f1_score(yt,pred,average='macro')),
            'temporal_macro_f1':float(f1_score(yt,smooth_pred,average='macro')),
            'frame_confusion_matrix':confusion_matrix(yt,pred,labels=model.classes_).tolist()}
    scaler,clf=model.steps[0][1],model.steps[1][1]
    metadata={'feature_version':FEATURE_VERSION,'procedure_hash':signature(p),'synthetic':synthetic,
        'algorithm':'StandardScaler + multinomial LogisticRegression','sklearn_training_version':sklearn.__version__,
        'python':platform.python_version(),'manifest_sha256':digest(manifest_path),'sample_hz':2,
        'model_id':digest(manifest_path)[:12], 'smoothing_frames':window, 'evaluation_status':report['evaluation_status'],
        'training_video_hashes':sorted(digest((manifest_path.parent/i['video']).resolve()) for i in data['sessions'] if i['split']=='train')}
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix('.tmp.npz')
    np.savez_compressed(temp,mean=scaler.mean_,scale=scaler.scale_,coef=clf.coef_,intercept=clf.intercept_,
                        classes=model.classes_,metadata=np.array(json.dumps(metadata)))
    temp.replace(output)
    output.with_suffix('.metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--output',default=str(ROOT/'models/lego_real.npz'))
    args=parser.parse_args(); print(json.dumps(train(args.manifest,args.output),indent=2))
