from pathlib import Path
import json
import html
import time
import cv2
import numpy as np
from stepguard_sprint1 import write_json
from stepguard_ml import sample_video,features,smooth_probabilities,load_model,sha256
from stepguard_rules import validate_procedure,ProcedureEngine,decide


def read_component_evidence(path,video_hash,duration):
    if not path:return []
    doc=json.loads(Path(path).read_text())
    if doc.get('video_sha256')!=video_hash:raise ValueError('Component evidence belongs to another video.')
    rows=doc.get('frames',[]);last=-1
    for r in rows:
        t=r['timestamp_seconds']
        if not isinstance(t,(int,float)) or not np.isfinite(t) or not last<t<=duration or t<0:
            raise ValueError('Invalid component evidence timestamps.')
        last=t
        for d in r.get('detections',[]):
            if not isinstance(d.get('class_name'),str) or not np.isfinite(d.get('score',float('nan'))) or not 0<=d['score']<=1:
                raise ValueError('Invalid component detection.')
    return rows


def convert_sprint1_evidence(predictions_path,video_path,output_path):
    """Wrap Sprint 1 output. Coverage remains FALSE until a real human review."""
    rows=json.loads(Path(predictions_path).read_text())
    result={'video_sha256':sha256(video_path),'source':'Sprint 1 detector; coverage not reviewed',
       'frames':[dict(r,coverage_reviewed=False,checkpoint_step=None) for r in rows]}
    write_json(output_path,result)
    return output_path


def analyse_video(video_path,model_path,procedure_path,output_dir,*,component_path=None,
                  threshold=.65,margin=.15,stable_frames=2,min_blur=5.,min_brightness=15.,max_brightness=245.):
    video_path=Path(video_path);out=Path(output_dir)
    p=validate_procedure(json.loads(Path(procedure_path).read_text()))
    model=load_model(model_path,p)
    frames,meta=sample_video(video_path,model['sample_hz'])
    if component_path and json.loads(Path(component_path).read_text()).get('synthetic') and not model['synthetic']:
        raise ValueError('Do not combine scripted synthetic component evidence with a real model.')
    evidence=read_component_evidence(component_path,sha256(video_path),meta['duration_seconds'])
    out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True)
    start=time.perf_counter()
    probs=model['model'].predict_proba(np.stack([features(f['image']) for f in frames]))
    temporal=smooth_probabilities(probs,model['window'])
    engine=ProcedureEngine(p,stable_frames)
    rows=[]
    # Exact timestamp association prevents reusing one evidence frame many times.
    component_map={round(e['timestamp_seconds'],6):e for e in evidence}
    for i,f in enumerate(frames):
        gray=cv2.cvtColor(f['image'],cv2.COLOR_BGR2GRAY)
        blur=float(cv2.Laplacian(gray,cv2.CV_64F).var());bright=float(gray.mean())
        quality_ok=blur>=min_blur and min_brightness<=bright<=max_brightness
        result=decide(temporal[i],model['classes'],threshold,margin,quality_ok)
        ev=component_map.get(round(f['timestamp_seconds'],6))
        state=engine.update(f['timestamp_seconds'],result['step_id'],ev)
        image_path=out/'frames'/f'frame_{f["frame_index"]:06d}.jpg'
        display=f['image'].copy()
        cv2.rectangle(display,(0,0),(320,27),(255,255,255),-1)
        cv2.putText(display,f'{f["timestamp_seconds"]:.2f}s {result["step_id"]} {result["score"]:.2f}',
                    (5,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
        cv2.imwrite(str(image_path),display)
        rows.append({'timestamp_seconds':f['timestamp_seconds'],'frame_index':f['frame_index'],
           **result,**state,'brightness':bright,'blur_variance':blur,
           'image':str(image_path.relative_to(out)),
           'guidance':next((s['instruction'] for s in p['steps'] if s['id']==result['step_id']),
                           'Cannot determine the step. Provide a clearer view.'),
           'evidence_source':ev.get('source','component evidence JSON') if ev else 'no component evidence'})
    unknown=sum(r['step_id']=='unknown' for r in rows)
    report={'project':'StepGuard','sprint':2,'dataset_type':'SYNTHETIC MODEL DEMO' if model['synthetic'] else 'REAL-DATA MODEL',
      'video_sha256':sha256(video_path),'procedure_id':p['id'],'procedure_version':p['version'],
      'model_version':model['model_version'],'policy':{'confidence':threshold,'margin':margin,'stable_frames':stable_frames,
        'min_blur':min_blur,'brightness_range':[min_brightness,max_brightness]},
      'sample_hz':model['sample_hz'],'window_frames':model['window'],
      'coverage':1-unknown/len(rows),'processed_frames':len(rows),'processing_seconds':time.perf_counter()-start,
      'component_evidence_frames':len(evidence),'component_frames_matched':sum(round(f['timestamp_seconds'],6) in component_map for f in frames),
      'states':engine.states,'alerts':engine.alerts,'timeline':rows,
      'limitations':['Observed checkpoint is not verified completion.',
        'Sequence alert means prerequisite was not observed; recordings may start mid-task.',
        'Component absence needs human-reviewed view coverage and may still be a missed detection.',
        'Confidence and quality thresholds are provisional, not calibrated.',
        'No exact pose, hidden-connection, safety or live-camera verification.']}
    write_json(out/'session_report.json',report)
    write_html(report,out/'session_report.html')
    return report


def write_html(report,path):
    esc=lambda s:html.escape(str(s))
    rows=''.join('<tr>'+''.join('<td>'+esc(r[k])+'</td>' for k in ['timestamp_seconds','step_id','state','component_check','score'])+
                 '<td><a href="'+esc(r['image'])+'">Evidence frame</a></td></tr>' for r in report['timeline'])
    alerts=''.join('<li>'+esc(f'{a["timestamp_seconds"]:.2f}s | {a["type"]} | {a["details"]} | {a["instruction"]} | {a["guide_ref"]}')+'</li>' for a in report['alerts'])
    text='<!doctype html><html><head><meta charset="utf-8"><title>StepGuard report</title><style>body{font-family:Arial;max-width:1100px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}th{background:#eef3f6}</style></head><body>'
    text+='<h1>StepGuard Sprint 2</h1><p>'+esc(report['dataset_type'])+'</p>'
    text+='<p>Model '+esc(report['model_version'])+'; procedure '+esc(report['procedure_id'])+' '+esc(report['procedure_version'])+'</p>'
    text+='<h2>Suspected alerts</h2><ul>'+alerts+'</ul><h2>Timeline</h2><table><tr><th>Seconds</th><th>Step</th><th>State</th><th>Components</th><th>Score</th><th>Image</th></tr>'+rows+'</table>'
    text+='<h2>Limitations</h2><ul>'+''.join('<li>'+esc(x)+'</li>' for x in report['limitations'])+'</ul></body></html>'
    Path(path).write_text(text,encoding='utf-8')
