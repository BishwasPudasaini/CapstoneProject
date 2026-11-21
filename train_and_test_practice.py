from pathlib import Path
import json
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from stepguard_ml import train_model, load_model, sample_video, features, label_at, smooth_probabilities
from stepguard_pipeline import analyse_video

ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
data = ROOT / 'practice_data'
work = ROOT / 'work' / 'practice'
procedure = json.loads((data / 'procedure.json').read_text())
model_path, validation = train_model(data / 'manifest.json', procedure, work / 'model', allow_synthetic=True)
print('SYNTHETIC ONLY: validation results are not real assembly accuracy.')
print('Training samples:', validation['train_samples'], 'Validation samples:', validation['validation_samples'])

# Freeze the model before examining held-out test recordings.
bundle = load_model(model_path, procedure)
manifest = json.loads((data / 'manifest.json').read_text())
truth, raw, temporal = [], [], []
for session in manifest['sessions']:
    if session['split'] != 'test':
        continue
    frames, _ = sample_video(data / session['video'], bundle['sample_hz'])
    probs = bundle['model'].predict_proba(np.stack([features(f['image']) for f in frames]))
    smooth = smooth_probabilities(probs, bundle['window'])
    for i, frame in enumerate(frames):
        label = label_at(session['segments'], frame['timestamp_seconds'])
        if label is None:
            continue
        truth.append(label)
        raw.append(bundle['classes'][int(probs[i].argmax())])
        temporal.append(bundle['classes'][int(smooth[i].argmax())])
test_results = {'dataset_type': 'SYNTHETIC ONLY', 'test_samples': len(truth),
                'classes': bundle['classes'], 'note': 'Closed-set step classification, before uncertainty gating. No real-world accuracy claim.'}
for name, prediction in [('frame_only', raw), ('temporal_average', temporal)]:
    test_results[name] = {'macro_f1': float(f1_score(truth, prediction, average='macro')),
                         'confusion_matrix': confusion_matrix(truth, prediction, labels=bundle['classes']).tolist()}
(work / 'held_out_test_results.json').write_text(json.dumps(test_results, indent=2))
print(json.dumps(test_results, indent=2))

correct = analyse_video(data / 'videos/test_correct.mp4', model_path,
                        data / 'procedure.json', work / 'correct_report')
reordered = analyse_video(data / 'videos/test_reordered.mp4', model_path,
                          data / 'procedure.json', work / 'reordered_report')
# This third run tests a rule with injected evidence; it does NOT run a component detector.
scripted = analyse_video(data / 'videos/test_reordered.mp4', model_path,
                         data / 'procedure.json', work / 'scripted_rule_report',
                         component_path=data / 'scripted_component_evidence.json')
assert not correct['alerts'], correct['alerts']
assert 'sequence_prerequisite_unobserved' in {a['type'] for a in reordered['alerts']}
assert 'visible_component_not_detected' in {a['type'] for a in scripted['alerts']}
print('PASS: correct order, reordered sequence, and separately injected component rule.')
print('Model:', model_path)
print('Reports:', work)
