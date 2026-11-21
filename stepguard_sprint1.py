"""Sprint 1: local video ingestion, frame manifests and reviewed YOLO dataset export.
No step recognition, error diagnosis or trained model is bundled.
"""
from pathlib import Path
import hashlib
import json
import math
import shutil
import uuid
import cv2
import numpy as np

EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
MAX_BYTES = 100 * 1024 * 1024


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def inspect_video(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Video does not exist: {path}')
    if path.suffix.lower() not in EXTENSIONS:
        raise ValueError('Use MP4, AVI, MOV or MKV.')
    if not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError('Video must be nonempty and at most 100 MiB.')
    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not cap.isOpened() or not ok or not np.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError('Cannot decode this video. Try an H.264 MP4 or MJPEG AVI.')
        duration = count / fps
        if duration > 300:
            raise ValueError('Sprint 1 accepts clips up to five minutes.')
        h, w = frame.shape[:2]
        if w * h > 3840 * 2160:
            raise ValueError('Use video resolution at or below 4K.')
        return {'fps': fps, 'frame_count': count, 'width': w, 'height': h,
                'duration_seconds': duration, 'bytes': path.stat().st_size,
                'timestamp_method': 'frame_index / FPS; constant-frame-rate input required'}
    finally:
        cap.release()


def ingest_video(source, root, *, sample_hz=1.0, session_id=None, split='unassigned',
                 assembler='unknown', synthetic=False, max_frames=300):
    """Decode locally and sample sequentially; split is assigned to the whole recording."""
    if split not in {'train', 'val', 'test', 'unassigned'}:
        raise ValueError('Invalid split.')
    if not 0 < sample_hz <= 10 or not 1 <= max_frames <= 300:
        raise ValueError('Use sample_hz in (0,10] and max_frames in [1,300].')
    source, root = Path(source), Path(root).resolve()
    meta = inspect_video(source)
    sid = session_id or uuid.uuid4().hex[:12]
    if not sid or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in sid):
        raise ValueError('Session ID may contain letters, digits, underscores and hyphens only.')
    destination = root / 'sessions' / sid
    if destination.exists():
        raise ValueError('Session already exists; use a new session ID.')
    destination.mkdir(parents=True)
    frames_dir = destination / 'frames'
    frames_dir.mkdir()
    video_path = destination / ('source' + source.suffix.lower())
    shutil.copy2(source, video_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cap = cv2.VideoCapture(str(video_path))
    records, idx, next_sample = [], 0, 0.0
    interval = 1.0 / sample_hz
    try:
        while len(records) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = idx / meta['fps']
            if timestamp + 1e-8 >= next_sample:
                name = f'frame_{idx:06d}.jpg'
                target = frames_dir / name
                if not cv2.imwrite(str(target), frame):
                    raise ValueError('Failed to write extracted image.')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                records.append({'session_id': sid, 'split': split,
                    'image': str(target.relative_to(root)), 'frame_index': idx,
                    'timestamp_seconds': round(timestamp, 6),
                    'width': frame.shape[1], 'height': frame.shape[0],
                    'blur_variance': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                    'mean_brightness': float(gray.mean()),
                    'reviewed': False, 'boxes': []})
                next_sample += interval
            idx += 1
    finally:
        cap.release()
    if not records:
        raise ValueError('No frames decoded.')
    session = {'session_id': sid, 'assembler': assembler, 'split': split,
               'synthetic': bool(synthetic), 'sha256': digest, 'metadata': meta,
               'sample_hz': sample_hz, 'sample_count': len(records),
               'source': str(video_path.relative_to(root))}
    write_json(destination / 'session.json', session)
    write_json(destination / 'annotations.json', records)
    return session, records


def make_fixture(path):
    """Generated moving shapes: tests video plumbing only; NOT assembly evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (320, 240))
    if not writer.isOpened():
        raise RuntimeError('MJPEG encoder unavailable.')
    try:
        for i in range(40):
            frame = np.full((240, 320, 3), 235, dtype=np.uint8)
            cv2.putText(frame, 'SYNTHETIC PIPELINE TEST', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1)
            cv2.rectangle(frame, (40+i, 80), (130+i, 150), (20, 60, 220), -1)
            cv2.circle(frame, (210, 150), 25, (220, 80, 20), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def validate_box(box, width, height, nclasses):
    if not isinstance(box.get('class_id'), int) or not 0 <= box['class_id'] < nclasses:
        raise ValueError('Unknown class_id.')
    xyxy = box.get('xyxy', [])
    if len(xyxy) != 4 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in xyxy):
        raise ValueError('Box must have four finite pixel coordinates.')
    x1, y1, x2, y2 = xyxy
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('Invalid or out-of-bounds box.')
    return [box['class_id'], (x1+x2)/(2*width), (y1+y2)/(2*height),
            (x2-x1)/width, (y2-y1)/height]


def export_dataset(root, output, class_names):
    """Reject synthetic, unreviewed, duplicate-session and malformed data.
    Split assignment lives in session.json; annotations must agree.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError('Class names must be nonempty and unique.')
    plans, seen_hashes, seen_images, sessions = [], {}, set(), []
    per_split = {k: {'sessions': 0, 'frames': 0, 'boxes': 0} for k in ('train','val','test')}
    class_counts = {k: [0]*len(class_names) for k in per_split}
    for session_file in sorted((root / 'sessions').glob('*/session.json')):
        s = json.loads(session_file.read_text())
        if s.get('synthetic') or s['split'] == 'unassigned':
            continue
        split, sid = s['split'], s['session_id']
        if split not in per_split:
            raise ValueError('Unknown session split.')
        if s['sha256'] in seen_hashes:
            raise ValueError('Duplicate source video detected. Do not duplicate sessions across splits.')
        seen_hashes[s['sha256']] = split
        rows = json.loads((session_file.parent / 'annotations.json').read_text())
        if not rows:
            raise ValueError(f'Empty session {sid}')
        per_split[split]['sessions'] += 1
        sessions.append({'session_id':sid,'split':split,'assembler':s['assembler'],'sha256':s['sha256']})
        for r in rows:
            if r['session_id'] != sid or r['split'] != split:
                raise ValueError('Frame/session split mismatch: split recordings, not frames.')
            if r.get('reviewed') is not True:
                raise ValueError(f'Annotation not reviewed: {r["image"]}')
            src = (root / r['image']).resolve()
            if not src.is_relative_to(root) or not src.is_file():
                raise ValueError('Image missing or outside dataset root.')
            img = cv2.imread(str(src))
            if img is None or img.shape[:2] != (r['height'], r['width']):
                raise ValueError('Image dimensions mismatch.')
            image_hash = hashlib.sha256(src.read_bytes()).hexdigest()
            if image_hash in seen_images:
                raise ValueError('Duplicate extracted image: deduplicate before training.')
            seen_images.add(image_hash)
            labels = [validate_box(b,r['width'],r['height'],len(class_names)) for b in r['boxes']]
            for label in labels:
                class_counts[split][label[0]] += 1
            per_split[split]['frames'] += 1
            per_split[split]['boxes'] += len(labels)
            plans.append((src, split, f'{sid}_{src.stem}', labels))
    for split in ('train','val'):
        if not per_split[split]['sessions'] or not per_split[split]['boxes']:
            raise ValueError(f'Need reviewed REAL {split} sessions with labelled boxes. Synthetic fixture is excluded.')
        if any(n == 0 for n in class_counts[split]):
            raise ValueError(f'Every selected class needs labelled examples in {split}.')
    if output.exists():
        raise ValueError('Export directory exists. Use a new version name to avoid stale labels.')
    for src, split, name, labels in plans:
        image_dir, label_dir = output/'images'/split, output/'labels'/split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, image_dir/(name+'.jpg'))
        text = '\n'.join(str(int(v[0]))+' '+' '.join(f'{x:.8f}' for x in v[1:]) for v in labels)
        (label_dir/(name+'.txt')).write_text(text, encoding='utf-8')
    # JSON is a valid YAML subset; Ultralytics reads this file through YAML.
    config = {'path':str(output),'train':'images/train','val':'images/val','names':class_names}
    if per_split['test']['frames']:
        config['test'] = 'images/test'
    write_json(output/'data.yaml',config)
    audit = {'counts':per_split, 'class_counts':class_counts, 'sessions':sessions,
             'note':'Minimum validity checks only, not proof of sample adequacy or independence.'}
    write_json(output/'split_audit.json',audit)
    return output/'data.yaml', audit


def predict_frames(weights, records, root, output, conf=0.25):
    """Run only a user-provided custom checkpoint; return timestamped component boxes."""
    from ultralytics import YOLO
    import time
    weights = Path(weights)
    if not weights.is_file():
        raise ValueError('Provide a trained custom checkpoint; none is bundled.')
    model = YOLO(str(weights))
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    events = []
    for r in records:
        start = time.perf_counter()
        prediction = model.predict(str(root/r['image']), conf=conf, verbose=False)[0]
        elapsed = (time.perf_counter()-start)*1000
        annotated = output/Path(r['image']).name
        cv2.imwrite(str(annotated),prediction.plot())
        boxes = [{'class_name':model.names[int(b.cls.item())],
                  'score':float(b.conf.item()), 'xyxy':b.xyxy[0].tolist()}
                 for b in prediction.boxes]
        events.append({'timestamp_seconds':r['timestamp_seconds'], 'frame_index':r['frame_index'],
                       'detections':boxes,'processing_ms':elapsed,'annotated_image':str(annotated)})
    write_json(output/'component_predictions.json',events)
    return events
