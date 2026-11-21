import unittest,copy,tempfile,json
from pathlib import Path
import numpy as np
from demo_data import PROCEDURE
from stepguard_rules import ProcedureEngine,validate_procedure,decide
from stepguard_ml import smooth_probabilities,validate_segments
from stepguard_pipeline import read_component_evidence

class Sprint2Tests(unittest.TestCase):
    def test_correct_order_no_alerts(self):
        e=ProcedureEngine(PROCEDURE)
        for i,s in enumerate(['S1','S1','S2','S2','S3','S3']):e.update(i,s)
        self.assertEqual(e.alerts,[])
        self.assertEqual(e.states['S3'],'observed_checkpoint')

    def test_reordered_and_duplicate_suppression(self):
        e=ProcedureEngine(PROCEDURE)
        for i,s in enumerate(['S1','S1','S3','S3','S3','S3']):e.update(i,s)
        self.assertEqual(len(e.alerts),1)
        self.assertEqual(e.alerts[0]['details'],['S2'])
        self.assertNotIn('S3',e.observed)

    def test_unknown_breaks_stability(self):
        e=ProcedureEngine(PROCEDURE)
        for i,s in enumerate(['S1','unknown','S1']):e.update(i,s)
        self.assertFalse(e.observed)
        e.update(3,'S1');self.assertIn('S1',e.observed)

    def test_no_coverage_no_component_claim(self):
        e=ProcedureEngine(PROCEDURE)
        for i,s in enumerate(['S1','S1','S2','S2','S3','S3','S3']):
            e.update(i,s,{'checkpoint_step':'S3','coverage_reviewed':False,'detections':[]})
        self.assertEqual(e.alerts,[])

    def test_missing_requires_sustained_reviewed_evidence(self):
        e=ProcedureEngine(PROCEDURE)
        ev={'checkpoint_step':'S3','coverage_reviewed':True,'detections':[{'class_name':'wheel','score':.9}]}
        for i,s in enumerate(['S1','S1','S2','S2','S3','S3','S3']):e.update(i,s,ev)
        self.assertEqual(e.alerts[-1]['type'],'visible_component_not_detected')
        self.assertIn('guide_ref',e.alerts[-1])

    def test_low_confidence_and_quality(self):
        self.assertEqual(decide([.4,.35,.25],['S1','S2','S3'])['step_id'],'unknown')
        self.assertEqual(decide([.99,.005,.005],['S1','S2','S3'],quality_ok=False)['step_id'],'unknown')
        self.assertEqual(decide([.9,.05,.05],['S1','S2','S3'])['step_id'],'S1')

    def test_cycle_and_bad_labels(self):
        p=copy.deepcopy(PROCEDURE);p['steps'][0]['requires']=['S3']
        with self.assertRaisesRegex(ValueError,'cycle'):validate_procedure(p)
        with self.assertRaises(ValueError):validate_segments([{'start':0,'end':3,'step_id':'bad'}],4,{'S1'})
        with self.assertRaises(ValueError):validate_segments([{'start':0,'end':3,'step_id':'S1'},{'start':2,'end':4,'step_id':'S1'}],4,{'S1'})

    def test_causal_averaging(self):
        p=np.array([[1.,0],[0.,1],[0.,1]])
        q=smooth_probabilities(p,2)
        np.testing.assert_allclose(q,[[1,0],[.5,.5],[0,1]])

    def test_source_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'e.json';p.write_text(json.dumps({'video_sha256':'wrong','frames':[]}))
            with self.assertRaises(ValueError):read_component_evidence(p,'correct',10)

    def test_monotonic_times(self):
        e=ProcedureEngine(PROCEDURE);e.update(1,'unknown')
        with self.assertRaises(ValueError):e.update(1,'unknown')

if __name__=='__main__':unittest.main()
