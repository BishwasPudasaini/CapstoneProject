"""Local single-user StepGuard application. Run: python -m streamlit run app.py"""
from pathlib import Path
import json, tempfile, hashlib, uuid
import streamlit as st
from engine import ROOT, procedure, Model, LiveMonitor, analyse_video
st.set_page_config(page_title='StepGuard',layout='wide')
st.title('StepGuard • Assembly Monitoring')
st.caption('Supported checkpoints: join red → blue → yellow → green as in your uploaded recordings.')
p=procedure(); model_path=ROOT/'models/lego_real.npz'
worker=st.session_state.get('worker'); running=worker is not None and worker.thread.is_alive()
mode=st.sidebar.selectbox('Detection method',['Colour + position check','Original ML baseline'],disabled=running)
# st.sidebar.caption('Colour + position uses fixed computer-vision rules for this four-colour build. The ML baseline does not verify arrangement.')
left,right=st.sidebar.slider('Work area: left / right (%)',0,100,(0,100),disabled=running)
top,bottom=st.sidebar.slider('Work area: top / bottom (%)',0,100,(0,100),disabled=running)
if right-left<10 or bottom-top<10:
    st.error('Select a work area at least 10% wide and high.');st.stop()
threshold=st.sidebar.slider('Minimum model score',.50,.99,.70,.01,disabled=running or mode=='Colour + position check')
st.sidebar.caption('A score is not a calibrated probability. Select thresholds using validation data.')
for s in p['steps']: st.sidebar.write(f"{s['id']} — {s['name']}")
model=None
if model_path.exists():
    try:
        model=Model(model_path,p)
        model.mode='colour_geometry' if mode=='Colour + position check' else 'ml'
        model.roi=(left/100,top/100,right/100,bottom/100)
    except Exception as exc: st.error(f'Model could not be loaded: {exc}')
if model is None: st.info('First upload and label your recordings in Train model. No pretrained weights are included.')
else:
    st.success('Trained model loaded')
    if model.meta.get('evaluation_status')=='not_evaluated': st.warning('')
    if model.meta['synthetic']: st.warning('This model was trained on generated footage. Physical webcam monitoring requires training on real recordings.')

def show_report(report,key):
    if report['timeline']:
        latest=report['timeline'][-1]
        st.write('Current visible state:', latest['step'], '—', latest.get('arrangement','not checked'))
        st.caption(latest.get('reason',''))
    st.caption('The table records previously confirmed states; it does not describe only the current frame.')
    cols=st.columns(3)
    cols[0].metric('Observed',len(report['steps'])-len(report['steps_not_observed']))
    cols[1].metric('Not observed',len(report['steps_not_observed']))
    cols[2].metric('Order warnings',len(report['alerts']))
    st.dataframe(report['steps'],hide_index=True)
    for event in report['alerts']: st.warning(event['message'])
    if report['finished']:
        if report['steps_not_observed']: st.warning('Review these unobserved checkpoints: '+', '.join(report['steps_not_observed']))
        elif report['alerts']: st.warning('All checkpoints were seen, but sequence warnings require review.')
        else: st.success('All four checkpoints observed in expected order.')
    st.caption('Unobserved checkpoints may reflect skipped steps, occlusion or an incomplete video. Visible states do not prove secure connections.')
    with st.expander('Frame predictions'): st.dataframe(report['timeline'][-600:],hide_index=True)
    st.download_button('Download report',json.dumps(report,indent=2),'stepguard_report.json','application/json',key=key)

training,recorded,live=st.tabs(['1. Train model','2. Recorded video','3. Live camera'])
with training:
    st.info('Trained model is already included. Open Recorded video or Live camera to try it.')
    training_only=st.checkbox('Training only',value=True,disabled=running)
    st.write('Training-only mode needs two distinct recordings. Full evaluation needs two training, one validation and one test recording.')
    presets=json.loads((ROOT/'label_presets.json').read_text()) if (ROOT/'label_presets.json').exists() else {}
    # st.caption('Record complete builds plus empty-table, occluded and incorrect-build views. Label only intervals you have checked. Leave movement intervals unlabelled. Do not split copies or clips of one original recording across partitions.')
    origin=st.radio('Recording type',['Real camera recordings','Generated footage'],disabled=running)
    uploads=st.file_uploader('Training recordings (up to 12 files, 100 MB and five minutes each)',type=['mp4','avi','mov','mkv'],accept_multiple_files=True,disabled=running,key='training_upload')
    entries=[]
    if len(uploads)>12: st.error('Please select at most 12 videos.'); uploads=[]
    unique=[]; seen_uploads=set()
    for u in uploads:
        h=hashlib.sha256(u.getvalue()).hexdigest()
        if h in seen_uploads:
            st.warning(f'{u.name}: identical to another uploaded video. Excluded; it cannot be an independent validation/test recording.')
        else: unique.append(u); seen_uploads.add(h)
    uploads=unique
    for i,u in enumerate(uploads):
        key=str(i)+'_'+hashlib.sha256(u.getvalue()).hexdigest()[:16]
        preset=presets.get(hashlib.sha256(u.getvalue()).hexdigest())
        with st.expander(f'{i+1}. {u.name}',expanded=True):
            st.video(u)
            split=st.selectbox('Partition',['train','validation','test'],index=0 if training_only or i<2 else min(i-1,2),key='split'+key+str(training_only),disabled=running or training_only)
            group=st.text_input('Original recording group (same group for clips from one recording)',value=preset['source_group'] if preset else f'recording_{i+1}',key='group'+key,disabled=running)
            st.caption('Times are seconds from the start. Replace all zero end times; add rows if needed. Unknown = empty, occluded or visibly incorrect examples.')
            rows=st.data_editor(preset['segments'] if preset else [{'start':0.,'end':0.,'step_id':x} for x in ['S1','S2','S3','S4','unknown']],num_rows='dynamic',key='labels'+key,disabled=running,column_config={'step_id':st.column_config.SelectboxColumn(options=['S1','S2','S3','S4','unknown'],required=True)})
            entries.append((u,split,group,rows))
    confirmed=st.checkbox('I checked the labels, have permission to use these videos.',disabled=running)
    if st.button('Train model' if training_only else 'Train and evaluate',disabled=running or not entries):
        try:
            if not confirmed: raise ValueError('Check the labels and confirm the dataset statement first.')
            if len(entries)<(2 if training_only else 4): raise ValueError('Provide two distinct recordings for training only, or four for full evaluation.')
            if any(u.size>100*1024*1024 for u,_,_,_ in entries): raise ValueError('Each video must be at most 100 MB.')
            folder=ROOT/'data'/uuid.uuid4().hex; folder.mkdir(parents=True)
            manifest={'synthetic':origin=='Generated footage','include_unknown':True,'training_only':training_only,'smoothing_frames':1,'sessions':[]}
            for i,(u,split,group,rows) in enumerate(entries):
                if not group.strip(): raise ValueError('Every recording needs a source group.')
                name=f'video_{i:02d}'+Path(u.name).suffix.lower(); (folder/name).write_bytes(u.getvalue())
                manifest['sessions'].append({'session_id':str(i),'source_group':group.strip(),'split':split,'video':name,'permission_confirmed':True,'segments':sorted(rows,key=lambda r:float(r['start']))})
            path=folder/'manifest.json';path.write_text(json.dumps(manifest,indent=2))
            with st.spinner('Training on labelled frames and evaluating held-out recordings…'):
                from train import train
                metrics=train(path,model_path)
            st.session_state.pop('recorded_report',None)
            st.session_state['trained_message']='Training completed. Open Recorded video. Check the evaluation status below before reporting any accuracy.'
            st.rerun()
        except Exception as exc: st.error(f'Training failed: {exc}')
    if 'trained_message' in st.session_state: st.success(st.session_state['trained_message'])
    metrics_path=model_path.with_suffix('.metrics.json')
    if metrics_path.exists():
        st.subheader('Training report')
        metrics=json.loads(metrics_path.read_text()); st.json(metrics,expanded=False)
        # st.caption('Frame F1 measures state recognition on labelled intervals. It does not measure physical assembly correctness or missing-step alert precision. Do not tune using the test partition.')
        st.download_button('Download trained model',model_path.read_bytes(),'lego_real.npz')
    with st.expander('Import a model trained on another approved computer'):
        imported=st.file_uploader('StepGuard NPZ model',type=['npz'],disabled=running)
        if st.button('Install model',disabled=imported is None or running):
            try:
                model_path.parent.mkdir(exist_ok=True)
                temp=model_path.with_suffix('.import.npz'); temp.write_bytes(imported.getvalue())
                check=Model(temp,p)
                if 'unknown' not in check.classes: raise ValueError('Model must include the unknown class.')
                temp.replace(model_path); metrics_path.unlink(missing_ok=True)
                st.session_state.pop('recorded_report',None); st.rerun()
            except Exception as exc: st.error(str(exc))
with recorded:
    video=st.file_uploader('Assembly video (maximum 100 MB, five minutes)',type=['mp4','mov','avi','mkv'],disabled=running,key='test_upload')
    if video:
        st.video(video)
        if model and hashlib.sha256(video.getvalue()).hexdigest() in model.meta.get('training_video_hashes',[]):
            st.warning('This recording was used for training (possibly under another name). This is a replay check, not an independent test.')
    if st.button('Analyse assembly',disabled=model is None or video is None or running):
        st.session_state.pop('recorded_report',None)
        try:
            if video.size>100*1024*1024: raise ValueError('Video must be at most 100 MB.')
            preview=st.empty()
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/('input'+Path(video.name).suffix.lower());path.write_bytes(video.getvalue())
                with st.spinner('Analysing assembly states…'):
                    report=analyse_video(path,model,p,threshold,lambda frame,row:preview.image(frame,channels='BGR'))
            st.session_state['recorded_report']=report
        except Exception as exc: st.error(str(exc))
    if 'recorded_report' in st.session_state: show_report(st.session_state['recorded_report'],'recorded_report_download')
with live:
    st.write('Uses the webcam attached to the computer running this app. Keep the whole build visible. Use one straight four-colour build, even lighting and a plain background. Colour mode checks the order starting at red, horizontally or vertically. Crop out spare bricks with the work-area sliders. Accuracy is not yet measured.')
    index=st.number_input('Camera index',0,5,0,disabled=running)
    consent=st.checkbox('I have permission to use this camera.',disabled=running)
    st.caption('Frames are processed locally and not saved. Stop before closing the app. Automatic stop after five minutes or a browser heartbeat timeout.')
    a,b=st.columns(2)
    if a.button('Start camera',disabled=running or model is None):
        if not consent: st.error('Confirm camera permission.')
        elif model.meta['synthetic']: st.error('Train on real recordings before physical-camera monitoring.')
        else:
            if worker: worker.stop()
            st.session_state['worker']=LiveMonitor(int(index),model,p,threshold).start()
            st.session_state['live_end_refreshed']=False;st.rerun()
    if b.button('Stop camera',disabled=not running): worker.stop();st.rerun()
    @st.fragment(run_every=.5)
    def live_view():
        monitor=st.session_state.get('worker')
        if monitor is None: return
        monitor.heartbeat(); frame,report,error=monitor.snapshot()
        if error: st.error(error)
        if frame is not None: st.image(frame,channels='BGR')
        show_report(report,'live_report_download')
        if report['finished']:
            st.caption('Ended: '+str(report['end_reason']))
            if not st.session_state.get('live_end_refreshed',False):
                st.session_state['live_end_refreshed']=True;st.rerun()
    live_view()
