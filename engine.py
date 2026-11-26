"""Shared recorded/live inference. Pure NumPy checkpoint loading, no pickle."""
from pathlib import Path
from collections import deque
import hashlib
import json
import time
import threading
import platform
import cv2
import numpy as np
from colour_detector import detect, annotate

ROOT = Path(__file__).resolve().parent
FEATURE_VERSION = 'lego-spatial-v1'

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()

def procedure(path=ROOT/'procedure.json'):
    p = json.loads(Path(path).read_text(encoding='utf-8'))
    steps = p['steps']; ids = [s['id'] for s in steps]
    if len(ids) < 3 or len(ids) != len(set(ids)) or 'unknown' in ids:
        raise ValueError('Use at least three unique step IDs, excluding unknown.')
    seen = set()
    for s in steps:
        if not s.get('name') or not s.get('instruction'):
            raise ValueError('Each step needs a name and reviewed instruction.')
        if not set(s.get('requires', [])).issubset(seen):
            raise ValueError('Prerequisites must refer to earlier steps in the list.')
        seen.add(s['id'])
    return p

def signature(p):
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()

def features(frame):
    image = cv2.resize(frame, (96, 96))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    parts = []
    for y in range(0,96,32):
        for x in range(0,96,32):
            hist = cv2.calcHist([hsv[y:y+32,x:x+32]], [0,1], None, [12,4], [0,180,0,256]).ravel()
            parts.extend(hist / (hist.sum()+1e-9))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    parts.extend(cv2.resize(gray,(16,16)).ravel()/255.)
    parts.extend(cv2.resize(cv2.Canny(gray,60,160),(16,16)).ravel()/255.)
    return np.asarray(parts, dtype=np.float64)

def quality(frame):
    gray = cv2.cvtColor(cv2.resize(frame,(320,240)), cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean()); blur = float(cv2.Laplacian(gray,cv2.CV_64F).var())
    return 15 <= brightness <= 245 and blur >= 5, brightness, blur

def video_frames(path, hz=2, max_seconds=300):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS); count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or not np.isfinite(fps) or fps<=0:
            raise ValueError('Video cannot be decoded. Try a constant-frame-rate MP4 or AVI.')
        if count > 0 and count/fps > max_seconds:
            raise ValueError(f'Video exceeds {max_seconds} seconds.')
        i = 0; target = 0.; emitted = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            t = i/fps
            if t > max_seconds: raise ValueError('Video exceeded the processing time limit.')
            if t + 1e-8 >= target:
                emitted += 1
                yield t, frame
                target += 1/hz
            i += 1
        if not emitted: raise ValueError('No readable frames found.')
        if count > 0 and i < count-2:
            raise ValueError('Video ended before the advertised frame count; re-encode or check corruption.')
    finally:
        cap.release()

class Model:
    def __init__(self, path, p):
        with np.load(path, allow_pickle=False) as z:
            self.mean=z['mean']; self.scale=z['scale']; self.coef=z['coef']; self.intercept=z['intercept']
            self.classes=z['classes'].tolist(); self.meta=json.loads(str(z['metadata'].item()))
        self.mode='colour_geometry'
        self.roi=(0.,0.,1.,1.)
        if self.meta['feature_version']!=FEATURE_VERSION or self.meta['procedure_hash']!=signature(p):
            raise ValueError('Model does not match this procedure/features. Train again.')
        if set(self.classes)-{'unknown'}!={s['id'] for s in p['steps']}:
            raise ValueError('Model step classes do not match procedure.')
        n=len(self.mean)
        if self.scale.shape!=(n,) or self.coef.shape!=(len(self.classes),n) or self.intercept.shape!=(len(self.classes),):
            raise ValueError('Invalid model dimensions.')
        if not all(np.isfinite(a).all() for a in (self.mean,self.scale,self.coef,self.intercept)) or (self.scale<=0).any():
            raise ValueError('Invalid model values.')
    def probabilities(self, frame):
        x=(features(frame)-self.mean)/self.scale
        logits=self.coef @ x + self.intercept
        exp=np.exp(logits-logits.max())
        return exp/exp.sum()

class Tracker:
    def __init__(self, p, stable=2, max_gap=1.5):
        self.p=p; self.stable=stable; self.max_gap=max_gap
        self.seen={}; self.events=[]; self.candidate=None; self.streak=0; self.last=None; self.warned=set()
    def update(self, t, step):
        if self.last is not None and t<=self.last: raise ValueError('Timestamps must increase.')
        if self.last is not None and t-self.last>self.max_gap:
            self.candidate=None; self.streak=0
        self.last=t
        if step=='unknown':
            self.candidate=None; self.streak=0; return
        allowed={s['id']:s for s in self.p['steps']}
        if step not in allowed: raise ValueError('Unsupported step ID.')
        self.streak=self.streak+1 if self.candidate==step else 1
        self.candidate=step
        if self.streak<self.stable: return
        missing=[x for x in allowed[step].get('requires',[]) if x not in self.seen]
        key=(step,tuple(missing))
        if missing and key not in self.warned:
            self.events.append({'time_seconds':round(t,3),'type':'possible_missing_or_out_of_order_step',
                'observed_step':step,'prerequisites_not_observed':missing,
                'message':f"Observed {step} before observing {', '.join(missing)}. Review the recording/build; a step may be skipped or outside the camera view.",
                'guidance':[allowed[x]['instruction'] for x in missing]})
            self.warned.add(key)
        self.seen.setdefault(step,round(t,3))
    def status(self):
        return [{'step':s['id'],'name':s['name'],'status':'observed' if s['id'] in self.seen else 'not observed',
                 'first_observed_seconds':self.seen.get(s['id'])} for s in self.p['steps']]

class Session:
    def __init__(self, model, p, threshold=.70, margin=.15):
        self.model=model; self.p=p; self.threshold=threshold; self.margin=margin
        self.history=deque(maxlen=int(model.meta.get('smoothing_frames',3))); self.tracker=Tracker(p, stable=3); self.rows=[]; self.last=None
        self.bad_streak=0; self.bad_active=False; self.arrangement_events=[]
    def process(self,t,frame):
        if self.last is not None and t-self.last>1.5: self.history.clear()
        self.last=t
        valid,brightness,blur=quality(frame)
        raw=self.model.probabilities(frame)
        if not valid: self.history.clear()
        else: self.history.append(raw)
        probs=np.mean(self.history,axis=0) if self.history else raw
        order=np.argsort(probs); idx=int(order[-1]); score=float(probs[idx]); margin=float(probs[idx]-probs[order[-2]])
        step=self.model.classes[idx] if valid and score>=self.threshold and margin>=self.margin else 'unknown'
        diagnostic={'ml_candidate':self.model.classes[idx], 'ml_score':round(score,4),
                    'ml_margin':round(margin,4)}
        canvas=frame.copy()
        if getattr(self.model,'mode','colour_geometry')=='colour_geometry':
            h,w=frame.shape[:2]
            l,top,r,bottom=getattr(self.model,'roi',(0.,0.,1.,1.))
            x1,y1,x2,y2=int(l*w),int(top*h),int(r*w),int(bottom*h)
            region=frame[y1:y2,x1:x2]
            result=detect(region)
            step=result['step'] if valid else 'unknown'
            if not valid:
                result.update(arrangement='uncertain',reason='Frame is too dark, bright or blurred.')
            self.bad_streak=self.bad_streak+1 if result['arrangement']=='incorrect_visible_order' and valid else 0
            if self.bad_streak>=3 and not self.bad_active:
                self.arrangement_events.append({'time_seconds':round(t,3),'type':'incorrect_visible_order',
                    'message':'Visible colour order differs from red → blue → yellow → green. Check the labelled regions.',
                    'colour_order':result['colour_order']})
                self.bad_active=True
            if self.bad_streak==0: self.bad_active=False
            canvas[y1:y2,x1:x2]=annotate(region,result)
            cv2.rectangle(canvas,(x1,y1),(x2-1,y2-1),(230,230,230),1)
            diagnostic.update(result)
            diagnostic['roi_pixels']=[x1,y1,x2,y2]
            label=f"{t:.1f}s  {step}  {result['arrangement']}"
        else:
            label=f'{t:.1f}s  {step}  ML score {score:.2f}'
            diagnostic['arrangement']='not_checked'
        self.tracker.update(t,step)
        row={'time_seconds':round(t,3),**diagnostic,'step':step,'quality_ok':bool(valid),
             'brightness':round(brightness,2),'blur':round(blur,2)}
        self.rows.append(row)
        image=cv2.resize(canvas,(640,480))
        cv2.rectangle(image,(0,0),(640,42),(20,20,20),-1)
        cv2.putText(image,label,(8,27),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
        return image,row
    def report(self, finished=False, reason=None):
        missing=[s['id'] for s in self.p['steps'] if s['id'] not in self.tracker.seen]
        return {'procedure':self.p['name'],'model':self.model.meta,'finished':finished,'end_reason':reason,
                'status':'steps_not_observed' if missing else 'all_checkpoints_observed',
                'steps_not_observed':missing,'steps':self.tracker.status(),'alerts':list(self.tracker.events)+list(self.arrangement_events),
                'timeline':list(self.rows),'policy':{'threshold':self.threshold,'margin':self.margin,'smoothing_frames':self.history.maxlen,'stable_samples':3,'detector':getattr(self.model,'mode','colour_geometry')},
                'limitations':['Not observed does not prove physically skipped.','Visible states are not proof of performed actions or secure connections.',
                'Scores and thresholds are not calibrated; unfamiliar objects can still be classified confidently.',
                'Colour regions can be confused with similarly coloured objects. Partial occlusion may resemble an earlier state.',
                'Colour geometry is rule-based, not learned; ML scores are diagnostic only in colour mode.',
                'No hidden-connection inspection or physical attachment verification.']}

def analyse_video(path,model,p,threshold=.70,progress=None):
    session=Session(model,p,threshold)
    for i,(t,frame) in enumerate(video_frames(path)):
        preview,row=session.process(t,frame)
        if progress and i%4==0: progress(preview,row)
    report=session.report(True,'end of recording'); report['video_sha256']=digest(path)
    return report

class LiveMonitor:
    """One capture worker. Reads continuously, analyses latest frames at 2 Hz.
    UI calls heartbeat(); idle browser sessions release camera after 15 seconds.
    source can be a local video path ONLY for integration testing/playback.
    """
    def __init__(self,source,model,p,threshold=.70,playback=False):
        self.source=source; self.session=Session(model,p,threshold); self.playback=playback
        self.lock=threading.Lock(); self.stop_event=threading.Event(); self.last_heartbeat=time.monotonic()
        self.preview=None; self.error=None; self.reason=None; self.finished=False
        self.thread=threading.Thread(target=self._run,daemon=True)
    def start(self): self.thread.start(); return self
    def heartbeat(self): self.last_heartbeat=time.monotonic()
    def stop(self):
        self.stop_event.set(); self.thread.join(timeout=3)
    def snapshot(self):
        with self.lock:
            return (None if self.preview is None else self.preview.copy(),self.session.report(self.finished,self.reason),self.error)
    def _run(self):
        cap=None
        try:
            backend=cv2.CAP_DSHOW if platform.system()=='Windows' and isinstance(self.source,int) else cv2.CAP_ANY
            cap=cv2.VideoCapture(self.source,backend)
            if not cap.isOpened(): raise ValueError('Cannot open camera. Close other camera apps, check permissions, or choose another camera index.')
            if not self.playback:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480); cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            fps=cap.get(cv2.CAP_PROP_FPS) if self.playback else 0
            if self.playback and (not np.isfinite(fps) or fps<=0): raise ValueError('Invalid playback FPS.')
            start=time.monotonic(); next_sample=0.; index=0
            while not self.stop_event.is_set():
                if time.monotonic()-self.last_heartbeat>15:
                    self.reason='browser heartbeat timeout'; break
                elapsed=time.monotonic()-start
                if elapsed>300:
                    self.reason='five-minute session limit'; break
                ok,frame=cap.read()
                if not ok:
                    if self.playback: self.reason='end of playback'; break
                    raise ValueError('Camera disconnected or frame read failed.')
                timestamp=index/fps if self.playback else time.monotonic()-start
                index+=1
                if timestamp+1e-8>=next_sample:
                    with self.lock: self.preview,_=self.session.process(timestamp,frame)
                    next_sample=timestamp+.5
                if self.playback: self.stop_event.wait(max(0,index/fps-(time.monotonic()-start)))
            if self.reason is None: self.reason='stopped by user'
        except Exception as exc:
            self.error=str(exc); self.reason='capture error'
        finally:
            if cap is not None: cap.release()
            with self.lock: self.finished=True
