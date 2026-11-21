from pathlib import Path
import json
from demo_data import create_demo
from stepguard_ml import train_model
from stepguard_pipeline import analyse_video


def run_demo(root='work/demo_run'):
    root=Path(root);data=create_demo(root/'data')
    procedure=json.loads((data/'procedure.json').read_text())
    model,metrics=train_model(data/'manifest.json',procedure,root/'model',allow_synthetic=True)
    report=analyse_video(data/'demo_reordered.avi',model,data/'procedure.json',root/'report',
      component_path=data/'component_rule_fixture.json')
    return root,metrics,report

if __name__=='__main__':
    root,metrics,report=run_demo()
    print('SYNTHETIC TEST ONLY — no real assembly accuracy claim.')
    print('Report:',root/'report'/'session_report.html')
    print('Alert types:',[a['type'] for a in report['alerts']])
