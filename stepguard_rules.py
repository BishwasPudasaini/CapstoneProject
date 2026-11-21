"""Explicit observable-checkpoint rules. No rule certifies assembly completion/safety."""
import math
from collections import deque
import numpy as np


def validate_procedure(p):
    if not p.get('id') or not p.get('version') or not p.get('steps'):
        raise ValueError('Procedure needs id, version and steps.')
    if not p.get('approved',False): raise ValueError('Review and approve the guide before analysis.')
    steps={s['id']:s for s in p['steps']}
    if len(steps)!=len(p['steps']) or 'unknown' in steps: raise ValueError('Duplicate/reserved step ID.')
    if len(steps)<2: raise ValueError('Use at least two supported step classes.')
    for s in steps.values():
        if not s.get('instruction') or not s.get('guide_ref'): raise ValueError('Every step needs guide text and reference.')
        for dep in s.get('requires',[]):
            if dep not in steps: raise ValueError('Unknown prerequisite.')
        for req in s.get('expected_components',[]):
            if not req.get('name') or not isinstance(req.get('min_count'),int) or req['min_count']<1:
                raise ValueError('Component rule needs name and positive min_count.')
    visiting=set();done=set()
    def walk(k):
        if k in visiting: raise ValueError('Procedure contains a cycle.')
        if k in done:return
        visiting.add(k)
        for dep in steps[k].get('requires',[]):walk(dep)
        visiting.remove(k);done.add(k)
    for k in steps:walk(k)
    return p


class ProcedureEngine:
    def __init__(self,procedure,stable_frames=2,component_score=.5):
        self.p=validate_procedure(procedure)
        if stable_frames<1:raise ValueError('stable_frames must be positive')
        self.steps={s['id']:s for s in procedure['steps']}
        self.stable_frames=stable_frames;self.component_score=component_score
        self.states={s:'not_observed' for s in self.steps}
        self.observed=set();self.alerts=[];self.emitted=set()
        self.candidate=None;self.streak=0;self.missing_streak={};self.previous_time=None

    def alert(self,kind,sid,t,details):
        key=(kind,sid,tuple(details))
        if key in self.emitted:return
        self.emitted.add(key)
        step=self.steps[sid]
        self.alerts.append({'type':kind,'step_id':sid,'timestamp_seconds':t,
          'details':list(details),'status':'suspected_requires_review','instruction':step['instruction'],
          'guide_ref':step['guide_ref'], 'guide_version':self.p['version']})

    def update(self,t,step_id,evidence=None):
        if not math.isfinite(t) or t<0 or (self.previous_time is not None and t<=self.previous_time):
            raise ValueError('Timestamps must be finite, nonnegative and strictly increasing.')
        self.previous_time=t
        if step_id is None or step_id=='unknown':
            self.candidate=None;self.streak=0;self.missing_streak.clear()
            return {'state':'unknown','component_check':'not_assessed'}
        if step_id not in self.steps:raise ValueError('Unknown predicted step')
        if step_id==self.candidate:self.streak+=1
        else:self.candidate=step_id;self.streak=1;self.missing_streak.clear()
        if self.streak<self.stable_frames:return {'state':'probable','component_check':'not_assessed'}
        step=self.steps[step_id]
        unmet=[s for s in step.get('requires',[]) if s not in self.observed]
        if unmet:
            self.states[step_id]='prerequisite_unobserved'
            self.alert('sequence_prerequisite_unobserved',step_id,t,unmet)
            # Absence of a recorded prerequisite is not proof of a physical omission.
            state='prerequisite_unobserved'
        else:
            self.observed.add(step_id);self.states[step_id]='observed_checkpoint'
            state='observed_checkpoint'
        component_check='not_assessed'
        required=step.get('expected_components',[])
        # A model failing to detect a part is not enough. A human-reviewed full checkpoint
        # view and explicit checkpoint ID are required for this prototype's component rule.
        if required and evidence and evidence.get('checkpoint_step')==step_id and evidence.get('coverage_reviewed') is True:
            component_check='checked_visible_view'
            missing=[]
            for req in required:
                count=sum(1 for d in evidence.get('detections',[]) if d['class_name']==req['name'] and d['score']>=self.component_score)
                key=(step_id,req['name'])
                self.missing_streak[key]=self.missing_streak.get(key,0)+1 if count<req['min_count'] else 0
                if self.missing_streak[key]>=self.stable_frames:missing.append(req['name'])
            if missing:
                component_check='suspected_missing_component'
                self.alert('visible_component_not_detected',step_id,t,missing)
        else:
            self.missing_streak.clear()
        return {'state':state,'component_check':component_check}


def decide(probabilities,classes,threshold=.65,margin=.15,quality_ok=True):
    p=np.asarray(probabilities,dtype=float)
    if len(p)!=len(classes) or not np.isfinite(p).all() or (p<0).any() or abs(p.sum()-1)>1e-4:
        raise ValueError('Invalid probabilities')
    if not 0<=threshold<=1 or not 0<=margin<=1:raise ValueError('Invalid decision policy')
    order=np.argsort(p);winner=int(order[-1]);gap=float(p[winner]-p[order[-2]]) if len(order)>1 else 1.
    reason='accepted'
    if not quality_ok:reason='image_quality'
    elif p[winner]<threshold:reason='low_confidence'
    elif gap<margin:reason='ambiguous_classes'
    return {'step_id':classes[winner] if reason=='accepted' else 'unknown',
            'top_candidate':classes[winner],'score':float(p[winner]),'margin':gap,'reason':reason}
