"""Procedural fixture generation, not a real assembly dataset or accuracy benchmark."""
from pathlib import Path
import json
import cv2
import numpy as np
from stepguard_sprint1 import write_json
from stepguard_ml import sha256

PROCEDURE={
 'id':'synthetic_cart','version':'demo-1','approved':True,'synthetic_demo':True,
 'kit_name':'Illustrated cart — synthetic fixture only',
 'steps':[
  {'id':'S1','name':'Base visible','requires':[], 'instruction':'Demo guide: place the blue base in view.',
   'guide_ref':'Authored synthetic guide / step 1','expected_components':[]},
  {'id':'S2','name':'First wheel visible','requires':['S1'],'instruction':'Demo guide: place the first wheel beside the base.',
   'guide_ref':'Authored synthetic guide / step 2','expected_components':[]},
  {'id':'S3','name':'Two wheels visible','requires':['S2'],'instruction':'Demo guide: inspect both wheel positions.',
   'guide_ref':'Authored synthetic guide / step 3','expected_components':[{'name':'wheel','min_count':2}]}
 ]}


def render_video(path,order,seed):
    rng=np.random.default_rng(seed);writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(320,240))
    if not writer.isOpened():raise RuntimeError('MJPEG encoder unavailable')
    segments=[];duration=4.
    dx=int(rng.integers(-5,6));dy=int(rng.integers(-4,5))
    try:
        for k,step in enumerate(order):
            segments.append({'start':k*duration,'end':(k+1)*duration,'step_id':step})
            for i in range(40):
                image=np.clip(np.full((240,320,3),225)+rng.normal(0,2,(240,320,3)),0,255).astype('uint8')
                cv2.putText(image,'SYNTHETIC TEST ONLY',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.45,(50,50,50),1)
                cv2.rectangle(image,(70+dx,80+dy),(250+dx,150+dy),(200,100,25),-1)
                if step in {'S2','S3'}:cv2.circle(image,(110+dx,160+dy),25,(35,35,35),-1)
                if step=='S3':cv2.circle(image,(210+dx,160+dy),25,(35,35,35),-1)
                writer.write(image)
    finally:writer.release()
    return segments


def create_demo(directory):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True);sessions=[]
    write_json(root/'procedure.json',PROCEDURE)
    for i in range(6):
        name=f'fixture_{i:02d}.avi';segments=render_video(root/name,['S1','S2','S3'],100+i)
        sessions.append({'id':f'fixture_{i:02d}','source_group':f'fixture_{i:02d}',
          'video':name,'split':'train' if i<4 else 'val','permission_confirmed':True,'segments':segments})
    write_json(root/'manifest.json',{'synthetic':True,'sessions':sessions})
    demo=root/'demo_reordered.avi'
    render_video(demo,['S1','S3','S2','S3'],999)
    # Scripted evidence exercises the rule engine independently of image recognition.
    # It intentionally reports one wheel late in the clip. This is NOT YOLO output.
    evidence=[]
    for t in np.arange(0,16,.5):
        if t>=12:
            evidence.append({'timestamp_seconds':float(t),'checkpoint_step':'S3','coverage_reviewed':True,
               'source':'SCRIPTED rule-test fixture, not a detector or human review',
               'detections':[{'class_name':'wheel','score':.99}]})
    write_json(root/'component_rule_fixture.json',{'video_sha256':sha256(demo),'synthetic':True,
      'source':'SCRIPTED TEST ONLY; injected component observations','frames':evidence})
    return root

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='work/demo')
    args=parser.parse_args();print(create_demo(args.output))
