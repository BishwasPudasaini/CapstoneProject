from pathlib import Path
import tempfile,json,zipfile,io
import streamlit as st
from run_demo import run_demo
from stepguard_pipeline import analyse_video

st.set_page_config(page_title='StepGuard',layout='wide')
st.title('StepGuard')
# st.caption('Recorded-video step baseline, temporal smoothing, explicit rules and evidence reports.')
mode=st.radio('Choose workflow',['Synthetic demonstration','Your assembly recording'])

def clean_previous():
    old=st.session_state.pop('temp',None)
    if old:old.cleanup()
    st.session_state.pop('result',None)

def new_root():
    clean_previous();temp=tempfile.TemporaryDirectory(prefix='sg2_');st.session_state['temp']=temp
    return Path(temp.name)

if mode=='Synthetic demonstration':
    # st.warning('Generated illustrations and scripted component evidence test the software only. These are not real assembly results.')
    if st.button('Run full demo'):
        root=new_root()
        try:
            with st.spinner('Generating fixtures, training the small baseline and analysing...'):
                result_root,metrics,report=run_demo(root)
            st.session_state['result']=(report,str(result_root/'report'),metrics,str(result_root/'data'/'demo_reordered.avi'))
        except Exception as e:st.error(str(e))
else:
    st.info('First train a step model for your confirmed kit using the notebook or train.py.')
    video=st.file_uploader('Assembly video',type=['mp4','avi','mov','mkv'])
    model_path=st.text_input('Trusted local step_model.joblib path')
    procedure=st.file_uploader('Reviewed procedure JSON',type=['json'],key='proc')
    component=st.file_uploader('Optional component evidence JSON (matching source video)',type=['json'],key='comp')
    st.caption('Coverage review is required for missing-component checks. Without it, only step/sequence analysis runs.')
    permitted=st.checkbox('I have permission to use this video and trust the local model file.')
    confidence=st.slider('Provisional confidence threshold',0.,1.,.65,.05)
    if st.button('Analyse recording',disabled=not(video and model_path and procedure and permitted)):
        root=new_root()
        try:
            if video.size>100*1024*1024:raise ValueError('Video exceeds 100 MiB.')
            src=root/('input'+Path(video.name).suffix.lower());src.write_bytes(video.getvalue())
            proc=root/'procedure.json';proc.write_bytes(procedure.getvalue())
            comp=None
            if component:comp=root/'components.json';comp.write_bytes(component.getvalue())
            with st.spinner('Analysing recorded video...'):
                report=analyse_video(src,model_path,proc,root/'report',component_path=comp,threshold=confidence)
            st.session_state['result']=(report,str(root/'report'),None,str(src))
        except Exception as e:st.error(str(e))

if 'result' in st.session_state:
    report,folder,metrics,src=st.session_state['result'];folder=Path(folder)
    st.subheader(report['dataset_type'])
    if metrics:
        with st.expander('Synthetic validation metrics — NOT real performance'):st.json(metrics)
    a,b,c=st.columns(3)
    a.metric('Sampled frames',report['processed_frames']);b.metric('Suspected alerts',len(report['alerts']))
    c.metric('Accepted step coverage',f'{report["coverage"]:.0%}')
    # st.caption('Observed checkpoint does not mean correct or completed assembly. Scores are not calibrated.')
    st.write('Step states');st.json(report['states'])
    for alert in report['alerts']:
        st.warning(f'{alert["timestamp_seconds"]:.2f}s | {alert["type"]} | {alert["details"]}')
        st.write(alert['instruction']);st.caption(alert['guide_ref'])
    table=[{k:r[k] for k in ['timestamp_seconds','step_id','state','component_check','score','reason']} for r in report['timeline']]
    st.dataframe(table,use_container_width=True)
    idx=st.selectbox('Inspect timestamped evidence',range(len(table)),format_func=lambda i:f'{table[i]["timestamp_seconds"]:.2f}s — {table[i]["step_id"]}')
    st.image(str(folder/report['timeline'][idx]['image']))
    st.write(report['timeline'][idx]['guidance'])
    if Path(src).suffix.lower()=='.mp4':
        st.video(Path(src).read_bytes(),start_time=int(table[idx]['timestamp_seconds']))
    st.download_button('Download JSON report',json.dumps(report,indent=2),'session_report.json','application/json')
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(folder))
    st.download_button('Download HTML report with evidence images',buffer.getvalue(),'stepguard_report.zip','application/zip')
    if st.button('Delete processed session'):
        clean_previous();st.rerun()
