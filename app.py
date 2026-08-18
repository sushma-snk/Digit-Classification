import io, json, time
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import tensorflow as tf

st.set_page_config(page_title='Human vs AI — Digit Challenge', page_icon='🤖', layout='wide')

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');
html,body,[class*="css"]{font-family:"DM Sans",sans-serif}.stApp{background:radial-gradient(circle at 10% 5%,rgba(124,58,237,.14),transparent 25%),radial-gradient(circle at 90% 8%,rgba(6,182,212,.13),transparent 25%),#f8fafc}.block-container{max-width:1120px;padding-top:1.5rem}.hero{background:linear-gradient(135deg,#17132e,#4c1d95 55%,#0e7490);color:white;padding:2rem 2.2rem;border-radius:30px;margin-bottom:1.2rem;box-shadow:0 18px 45px rgba(30,20,70,.20)}.hero h1{font-family:"Space Grotesk";font-size:2.9rem;margin:0}.hero p{opacity:.86;margin:.35rem 0;font-size:1.05rem}.metric,.card{background:rgba(255,255,255,.94);border:1px solid #e5e7eb;border-radius:20px;padding:1rem;box-shadow:0 8px 25px rgba(15,23,42,.06)}.metric{text-align:center}.metric .number{font-family:"Space Grotesk";font-size:1.8rem;font-weight:800}.metric .label{color:#64748b;font-size:.82rem;font-weight:700}.prediction{background:white;border:2px solid #c4b5fd;border-radius:26px;padding:1.5rem;text-align:center}.big-digit{font-family:"Space Grotesk";font-size:5rem;line-height:1;font-weight:800}.success-card{background:#ecfdf5;border:2px solid #86efac;border-radius:22px;padding:1.25rem}.error-card{background:#fff1f2;border:2px solid #fda4af;border-radius:22px;padding:1.25rem}.info-card{background:#eff6ff;border:1px solid #bfdbfe;border-radius:20px;padding:1.1rem}.stButton>button{border-radius:15px!important;min-height:50px!important;font-weight:800!important}
</style>''', unsafe_allow_html=True)

BASE=Path(__file__).parent; MODEL_PATH=BASE/'models'/'mnist_cnn.keras'; CORRECTIONS_PATH=BASE/'data'/'corrections.json'
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists(): st.error('Model file is missing. Run train_model.py first and place models/mnist_cnn.keras here.'); st.stop()
    return tf.keras.models.load_model(MODEL_PATH)
model=load_model()

for k,v in {'prediction':None,'processed_image':None,'original_image':None,'feedback_given':False,'correct_count':0,'wrong_count':0,'corrections':[],'round':1,'message':None,'pending_training':None}.items():
    if k not in st.session_state: st.session_state[k]=v

def reset_attempt():
    for k in ['prediction','processed_image','original_image','pending_training']: st.session_state[k]=None
    st.session_state.feedback_given=False; st.session_state.message=None

def load_corrections():
    try:
        with open(CORRECTIONS_PATH,encoding='utf-8') as f:return json.load(f)
    except:return []
if not st.session_state.corrections: st.session_state.corrections=load_corrections()

def save_correction(predicted,actual):
    CORRECTIONS_PATH.parent.mkdir(exist_ok=True)
    st.session_state.corrections.append({'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),'predicted':int(predicted),'actual':int(actual)})
    with open(CORRECTIONS_PATH,'w',encoding='utf-8') as f: json.dump(st.session_state.corrections,f,indent=2)

def crop_to_content(img):
    gray=ImageOps.grayscale(img); arr=np.asarray(gray)
    mask=255-arr if arr.mean()>127 else arr
    threshold=max(20,float(np.percentile(mask,75)*.20)); ys,xs=np.where(mask>threshold)
    if len(xs)<20:return img
    l,r,t,b=xs.min(),xs.max(),ys.min(),ys.max(); pad=max(5,int(.12*max(r-l+1,b-t+1)))
    return img.crop((max(0,l-pad),max(0,t-pad),min(img.width-1,r+pad)+1,min(img.height-1,b+pad)+1))

def preprocess_digit(uploaded):
    img=Image.open(uploaded).convert('L'); img=ImageEnhance.Contrast(img).enhance(2.0).filter(ImageFilter.GaussianBlur(.4)); arr=np.asarray(img); 
    if arr.mean()>=100: arr=255-arr
    img=crop_to_content(Image.fromarray(arr.astype(np.uint8))); w,h=img.size; side=max(w,h); canvas=Image.new('L',(side,side),0); canvas.paste(img,((side-w)//2,(side-h)//2))
    bbox=canvas.getbbox()
    if bbox:
        content=canvas.crop(bbox); content.thumbnail((20,20),Image.Resampling.LANCZOS); final=Image.new('L',(28,28),0); final.paste(content,((28-content.width)//2,(28-content.height)//2))
    else: final=canvas.resize((28,28),Image.Resampling.LANCZOS)
    return ImageEnhance.Contrast(final).enhance(1.25).resize((28,28),Image.Resampling.LANCZOS)

def predict(img):
    x=np.asarray(img,dtype=np.float32).reshape(1,28,28,1)/255.; probs=model.predict(x,verbose=0)[0]; p=int(np.argmax(probs)); return p,float(probs[p]),probs

def image_from_upload(u): return Image.open(io.BytesIO(u.getvalue())).convert('RGB')

st.markdown('''<div class="hero"><div style="font-weight:800;opacity:.75;letter-spacing:.08em">DIGITAL FLUENCY • AI / MACHINE LEARNING</div><h1>🧑‍🎨 HUMAN vs 🤖 AI</h1><p>Write a digit. Let AI guess. Correct it. Then see whether the AI can learn.</p></div>''',unsafe_allow_html=True)

total=st.session_state.correct_count+st.session_state.wrong_count; accuracy=100*st.session_state.correct_count/total if total else 0
cols=st.columns(4)
for col,label,value in zip(cols,['🎯 ATTEMPTS','✅ AI CORRECT','❌ AI WRONG','📊 CLASS AI SCORE'],[total,st.session_state.correct_count,st.session_state.wrong_count,f'{accuracy:.1f}%']):
    with col: st.markdown(f'<div class="metric"><div class="label">{label}</div><div class="number">{value}</div></div>',unsafe_allow_html=True)
st.write('')

left,right=st.columns([1.1,1])
with left:
    st.markdown('### ✍️ Give AI a digit')
    mode=st.radio('Choose input method',['📤 Upload image','📷 Capture image'],horizontal=True,label_visibility='collapsed')
    uploaded=st.file_uploader('Upload a photo of one handwritten digit',type=['png','jpg','jpeg']) if mode.startswith('📤') else st.camera_input('Take a photo of one handwritten digit')
    if uploaded is not None:
        try:
            if st.session_state.original_image is None:
                st.session_state.original_image=image_from_upload(uploaded); st.session_state.processed_image=preprocess_digit(uploaded); st.session_state.prediction=None; st.session_state.feedback_given=False; st.session_state.message=None
            st.image(st.session_state.original_image,caption='Your digit',width=360)
            if st.button('🤖 ASK AI TO CLASSIFY',type='primary',use_container_width=True):
                p,c,probs=predict(st.session_state.processed_image); st.session_state.prediction={'digit':p,'confidence':c,'probs':probs.tolist()}; st.session_state.feedback_given=False; st.session_state.message=None; st.rerun()
        except Exception as e: st.error(f'Could not process this image: {e}')
with right:
    st.markdown('### 🤖 AI prediction')
    if st.session_state.prediction is None: st.markdown('<div class="info-card"><h3>AI is waiting...</h3><p>Upload or capture a handwritten digit, then ask the AI to classify it.</p></div>',unsafe_allow_html=True)
    else:
        pr=st.session_state.prediction; p=pr['digit']; c=pr['confidence']; probs=np.array(pr['probs'])
        st.markdown(f'<div class="prediction"><div style="color:#64748b;font-weight:800">🤖 AI THINKS THIS IS</div><div class="big-digit">{p}</div><div style="font-weight:800">Confidence: {c*100:.1f}%</div></div>',unsafe_allow_html=True); st.progress(c,text=f'AI confidence: {c*100:.1f}%')
        with st.expander('🔍 See all digit probabilities'):
            for d,x in enumerate(probs): st.write(f'**{d}** — {x*100:.2f}%'); st.progress(float(x))
        if not st.session_state.feedback_given:
            st.markdown('### Was AI right?'); b1,b2=st.columns(2)
            with b1:
                if st.button('✅ YES — AI IS RIGHT',use_container_width=True): st.session_state.correct_count+=1; st.session_state.feedback_given=True; st.session_state.message='correct'; st.rerun()
            with b2:
                if st.button('❌ NO — AI IS WRONG',use_container_width=True): st.session_state.wrong_count+=1; st.session_state.feedback_given=True; st.session_state.message='wrong'; st.rerun()

if st.session_state.feedback_given:
    st.divider()
    if st.session_state.message=='correct': st.markdown('<div class="success-card"><h2>🎉 AI GOT IT RIGHT!</h2><p>You confirmed the model prediction.</p></div>',unsafe_allow_html=True)
    elif st.session_state.message=='wrong':
        st.markdown('<div class="error-card"><h2>😵 AI GOT IT WRONG!</h2><p>Now you get to teach the AI what the digit actually is.</p></div>',unsafe_allow_html=True)
        actual=st.number_input('What is the actual digit?',0,9,0,1,key='actual_digit')
        if st.button('🧠 TEACH AI THIS CORRECTION',type='primary',use_container_width=True):
            save_correction(st.session_state.prediction['digit'],int(actual)); st.session_state.pending_training={'image':np.asarray(st.session_state.processed_image,dtype=np.float32)/255.,'label':int(actual)}; st.session_state.message='training'; st.rerun()
    elif st.session_state.message=='training':
        st.markdown('<div class="info-card"><h2>🧠 NEW EXAMPLE RECEIVED</h2><p>The human supplied a label for an example AI got wrong. Fine-tune the running model with this labelled example.</p></div>',unsafe_allow_html=True)
        st.warning('The correction is stored locally. Fine-tuning changes the running model for this app session; the original MNIST model on disk is preserved.')
        if st.button('🔄 RETRAIN / FINE-TUNE AI',type='primary',use_container_width=True):
            item=st.session_state.pending_training; x=np.asarray(item['image'],dtype=np.float32).reshape(1,28,28,1); y=tf.keras.utils.to_categorical([item['label']],10)
            old=float(tf.keras.backend.get_value(model.optimizer.learning_rate)); tf.keras.backend.set_value(model.optimizer.learning_rate,1e-4)
            model.fit(x,y,epochs=3,batch_size=1,verbose=0); tf.keras.backend.set_value(model.optimizer.learning_rate,old); st.session_state.message='trained'; st.rerun()
    elif st.session_state.message=='trained': st.markdown('<div class="success-card"><h2>🧠 AI UPDATED!</h2><p>Your labelled correction was used for a small fine-tuning step. Try another handwritten digit.</p></div>',unsafe_allow_html=True)
    if st.button('➡️ NEW ATTEMPT',type='primary',use_container_width=True): st.session_state.round+=1; reset_attempt(); st.rerun()

st.divider(); st.markdown('## 🧠 What is happening behind the scenes?')
for col,title,text in zip(st.columns(4),['1️⃣ INPUT','2️⃣ PREDICT','3️⃣ FEEDBACK','4️⃣ LEARN'],['Your handwritten digit becomes an image.','The trained CNN finds patterns and predicts 0–9.','A human confirms or corrects the prediction.','A corrected example can be used for fine-tuning.']):
    with col: st.markdown(f'<div class="card"><h3>{title}</h3><p>{text}</p></div>',unsafe_allow_html=True)
st.caption(f'Round {st.session_state.round} • Stored human corrections: {len(st.session_state.corrections)}')
with st.expander('📚 Teacher explanation — use this after the activity'):
    st.markdown('''**Training data → Model → Prediction → Feedback → Updated model**\n\nThe original model was trained using MNIST. A student correction becomes a new labelled example. In this demo it is used for a small fine-tuning step on the running model; it is not full retraining from scratch.\n\nAsk: Why can two people write the same digit differently? Why might AI confuse 3 and 8? Does 95% confidence mean definitely correct? What if we gave it 1,000 corrected examples?''')
with st.expander('⚙️ Teacher controls'):
    if st.button('Reset classroom score'): st.session_state.correct_count=0; st.session_state.wrong_count=0; st.session_state.round=1; reset_attempt(); st.rerun()
