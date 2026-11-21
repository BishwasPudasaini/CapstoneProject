
from pathlib import Path
import hashlib
import json
import platform
import importlib.metadata
import cv2
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from stepguard_sprint1 import inspect_video, write_json

FEATURE_VERSION = 'hsv_grid_gray_edges_v1'


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def sample_video(path, hz=2., max_samples=600):
    meta=inspect_video(path)
    if not 0 < hz <= 10: raise ValueError('Sampling rate must be in (0,10].')
    if meta['duration_seconds']*hz>max_samples+1e-6:
        raise ValueError('Too many samples; reduce sampling rate or use a shorter clip.')
    cap=cv2.VideoCapture(str(path)); frames=[]; i=0; next_time=0.
    try:
        while len(frames)<max_samples:
            ok,img=cap.read()
            if not ok: break
            t=i/meta['fps']
            if t+1e-8 >= next_time:
               
                img=cv2.resize(img,(320,240))
                frames.append({'timestamp_seconds':round(t,6),'frame_index':i,'image':img})
                next_time+=1/hz
            i+=1
    finally: cap.release()
    if not frames: raise ValueError('No frames decoded.')
    return frames,meta


def features(image):
    small=cv2.resize(image,(96,72))
    hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    parts=[]
    for y in (0,36):
        for x in (0,48):
            hist=cv2.calcHist([hsv[y:y+36,x:x+48]],[0,1],None,[12,4],[0,180,0,256]).ravel()
            parts.extend(hist/(hist.sum()+1e-9))
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    parts.extend(cv2.resize(gray,(12,9)).ravel()/255.)
    edges=cv2.Canny(gray,60,140)
    parts.extend(cv2.resize(edges,(12,9)).ravel()/255.)
    return np.asarray(parts,dtype=np.float32)


def smooth_probabilities(probs,window=3):
    if not isinstance(window,int) or window<1: raise ValueError('window must be a positive integer')
    return np.asarray([probs[max(0,i-window+1):i+1].mean(axis=0) for i in range(len(probs))])


def validate_segments(segments,duration,allowed):
    last=0.
    for seg in sorted(segments,key=lambda x:x['start']):
        start,end=float(seg['start']),float(seg['end'])
        if not np.isfinite([start,end]).all() or not 0<=start<end<=duration+.1:
            raise ValueError('Invalid step interval.')
        if start < last-1e-6: raise ValueError('Overlapping step intervals.')
        if seg['step_id'] not in allowed: raise ValueError('Unknown step label.')
        last=end


def label_at(segments,t):
    for seg in segments:
        if seg['start']<=t<seg['end']: return seg['step_id']
    return None  # Unlabelled gaps are excluded; never silently assigned a step.


def train_model(manifest_path,procedure,output,*,allow_synthetic=False,hz=2.,window=3):
    from stepguard_rules import validate_procedure
    validate_procedure(procedure)
    manifest_path=Path(manifest_path).resolve(); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    doc=json.loads(manifest_path.read_text()); sessions=doc['sessions']
    if not sessions: raise ValueError('Empty training manifest.')
    synthetic=bool(doc.get('synthetic',False))
    if synthetic and not allow_synthetic: raise ValueError('Synthetic fixture requires explicit allow_synthetic=True.')
    if not synthetic and procedure.get('synthetic_demo',False): raise ValueError('Replace the demo procedure before real training.')
    ids=set(); hashes=set(); groups={}; data={'train':[],'val':[]}; audit=[]
    allowed={s['id'] for s in procedure['steps']}
    for s in sessions:
        if s['id'] in ids: raise ValueError('Duplicate session ID.')
        ids.add(s['id'])
        if s['split'] not in {'train','val','test'}: raise ValueError('Unknown split.')
        group=s.get('source_group',s['id'])
        if group in groups and groups[group]!=s['split']: raise ValueError('Source group crosses splits.')
        groups[group]=s['split']
        path=(manifest_path.parent/s['video']).resolve()
        digest=sha256(path)
        if digest in hashes: raise ValueError('Duplicate source recording; split independent sessions.')
        hashes.add(digest)
        audit.append({'session':s['id'],'group':group,'split':s['split'],'sha256':digest})
        if s['split']=='test': continue  # Never used for fitting, threshold selection or validation.
        if not synthetic and not s.get('permission_confirmed',False): raise ValueError('Confirm footage permission.')
        frames,meta=sample_video(path,hz)
        validate_segments(s['segments'],meta['duration_seconds'],allowed)
        keep=[(f,label_at(s['segments'],f['timestamp_seconds'])) for f in frames]
        labelled=[(f,label) for f,label in keep if label is not None]
        if not labelled: raise ValueError('No labelled samples in session '+s['id'])
        data[s['split']].append({'X':np.stack([features(f['image']) for f in frames]),
           'y':[label_at(s['segments'],f['timestamp_seconds']) for f in frames], 'id':s['id']})
    for split in ('train','val'):
        labels={l for s in data[split] for l in s['y'] if l is not None}
        if labels!=allowed: raise ValueError(f'{split} needs all procedure step classes. Got {labels}.')
    X=np.concatenate([s['X'][[l is not None for l in s['y']]] for s in data['train']])
    y=np.asarray([l for s in data['train'] for l in s['y'] if l is not None])
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42))
    model.fit(X,y); classes=model.classes_.tolist()
    truth=[]; raw=[]; temporal=[]
    for s in data['val']:
        probs=model.predict_proba(s['X']); smoothed=smooth_probabilities(probs,window)
        for j,label in enumerate(s['y']):
            if label is None: continue
            truth.append(label);raw.append(classes[int(probs[j].argmax())]);temporal.append(classes[int(smoothed[j].argmax())])
    scores={}
    for name,pred in [('frame_only',raw),('temporal_average',temporal)]:
        scores[name]={'macro_f1':float(f1_score(truth,pred,labels=classes,average='macro',zero_division=0)),
          'confusion_matrix':confusion_matrix(truth,pred,labels=classes).tolist(),
          'per_class':classification_report(truth,pred,labels=classes,output_dict=True,zero_division=0)}
    versions={name:importlib.metadata.version(name) for name in ['scikit-learn','numpy','opencv-python-headless','joblib']}
    bundle={'schema':1,'feature_version':FEATURE_VERSION,'model':model,'classes':classes,
      'procedure_id':procedure['id'],'procedure_version':procedure['version'],'synthetic':synthetic,
      'sample_hz':hz,'window':window,'versions':versions,'python':platform.python_version(),
      'model_version':hashlib.sha256((sha256(manifest_path)+str(hz)+str(window)+FEATURE_VERSION).encode()+model[-1].coef_.tobytes()).hexdigest()[:12]}
    joblib.dump(bundle,out/'step_model.joblib')
    report={'dataset_type':'SYNTHETIC TEST ONLY' if synthetic else 'REAL LABELLED FOOTAGE',
      'validation_samples':len(truth),'train_samples':len(y),'classes':classes,'metrics':scores,
      'sessions':audit,'note':'Validation scores, not final held-out results. Temporal averaging can worsen boundaries.'}
    write_json(out/'training_report.json',report)
    write_json(out/'environment.json',versions)
    return out/'step_model.joblib',report


def load_model(path,procedure):
  
    bundle=joblib.load(path)
    if bundle.get('feature_version')!=FEATURE_VERSION: raise ValueError('Feature-version mismatch; retrain.')
    if bundle['procedure_id']!=procedure['id'] or bundle['procedure_version']!=procedure['version']:
        raise ValueError('Model and procedure version do not match.')
    if set(bundle['classes'])!={s['id'] for s in procedure['steps']}: raise ValueError('Step labels do not match.')
    if bundle['versions']['scikit-learn']!=importlib.metadata.version('scikit-learn'):
        raise ValueError('scikit-learn version differs from training. Use recorded environment or retrain.')
    return bundle
