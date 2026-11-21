import argparse,json
from pathlib import Path
from stepguard_ml import train_model
from stepguard_rules import validate_procedure

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Train a Sprint 2 CPU step baseline.')
    p.add_argument('--manifest',required=True);p.add_argument('--procedure',required=True)
    p.add_argument('--output',default='work/model');p.add_argument('--allow-synthetic',action='store_true')
    p.add_argument('--hz',type=float,default=2.);p.add_argument('--window',type=int,default=3)
    a=p.parse_args()
    proc=validate_procedure(json.loads(Path(a.procedure).read_text()))
    model,report=train_model(a.manifest,proc,a.output,allow_synthetic=a.allow_synthetic,hz=a.hz,window=a.window)
    print('Model:',model);print(json.dumps(report,indent=2))
