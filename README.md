# StepGuard — colour and position update

## Run
1. Extract this ZIP to a new folder. Keep your old project as a backup.
2. Open this folder in VS Code and open its terminal.
3. Run `python -m pip install -r requirements.txt` if dependencies are missing.
4. Run `python -m streamlit run app.py`.
5. Select **Colour + position check** in the sidebar (default).
6. Open Live camera, confirm permission, and start. Use one straight build in a plain, evenly lit work area.
7. Show red, then red+blue, then red+blue+yellow, then all four with green at the end. Hold each stage for 3–5 seconds with hands clear.
8. Stop/start for a new build. The table remembers states from the current session.

No retraining is needed for colour mode. Your uploaded trained model is included unchanged. Original ML baseline remains selectable. Training updates only that ML baseline, NOT colour thresholds or geometric rules. Scikit-learn is only imported for training, not live or recorded inference.

## What changed
- colour_detector.py locates coloured regions using OpenCV HSV masks and shape filters.
- It checks that regions form a straight, approximately adjacent chain in red/blue/yellow/green order, starting from red. Rotation in the image is supported; strong perspective is not reliably handled.
- Labels and boxes show the evidence used. Unknown/uncertain and incorrect visible order are separate results.
- Both recorded and live modes use the same detector.
- Three consecutive accepted samples are required to record a stage. No earlier stages are filled in automatically.
- Work-area sliders crop distracting objects. Set them before starting; keep every brick inside the rectangle with some margin.
- Reports keep the original ML prediction as a diagnostic. A score of 1.0 does not certify correctness. Colour mode does not use the ML threshold slider.

## Validation and limits
Development checks, not independent accuracy measurements:
- Uploaded webcam video WIN_20261126_14_30_10_Pro.mp4: S1 at 2.530s, S2 at 9.012s, S3 at 17.019s, S4 at 27.002s, no sequence warnings.
- Latest completed-tower screenshot (camera area cropped): S4.
- Earlier wrong-order screenshot (camera area cropped): incorrect visible order; one edge-clipped colour is excluded.
- Older recording_2: all four checkpoints observed.
- Older recording_1: only S1 confirmed. Unequal brick geometry, occlusions and colour segmentation remain failure cases. Use baseline mode to compare, not to certify correct arrangement.
- Six unit tests passed (including four rotations of each stage, swapped colours, separated objects, blank view, duplicate colours, and no invented prerequisite steps).
- Streamlit initial page loaded without exceptions in AppTest.
- Physical Windows webcam capture was NOT tested here. Test a new live build on your computer.

This is a rule-based computer-vision addition, not a newly trained AI model. It is specific to one straight four-colour build. Similar colours in the background, shadows, hands, clipped bricks and reflections can confuse it. A hidden or undetected final brick can resemble an earlier stage. Coloured regions do not prove objects are LEGO or physically attached. A final correct view does not prove all actions happened in the right order.

The supplied videos/screenshots were used in development, so do not report these replay checks as held-out accuracy. Record new correct, skipped-step and wrong-order builds for evaluation. Preserve source recording separation if training the ML model.

## Files
app.py: Streamlit UI; engine.py: shared inference, tracking and capture; colour_detector.py: visual order rules; train.py: existing ML training; procedure.json: steps; models/: original uploaded weights and metrics; requirements.txt: dependencies; label_presets.json: previous known intervals. test_colour.py and validation.json document checks. webcam_check_report.json is the recorded-video replay result, not a live-camera result.

Run software checks with `python -m unittest test_colour.py`.
