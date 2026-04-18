import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image
import os
import tempfile
import math
import time
import base64

# ==========================================
# 1. KONFIGURASI
# ==========================================
# --- Path & Aset ---
MODEL_PATH = "Code/deepfake_detector_efficientnet.pth" 
LOCAL_ROBOT_ICON = "Image/robot1.png" 
LOCAL_IMAGE = "Image/AIBG.png" 

# --- Parameter Sistem ---
IMG_SIZE = (224, 224)
FRAMES_PER_VIDEO = 20         
FACE_LIMIT = 3                
MAX_FRAMES_TO_DISPLAY = 10 
FACE_PADDING = 0             
MTCNN_CONFIDENCE_THRESHOLD = 0.85  
MIN_FACE_SIZE = 20                 
FRAME_FAKE_THRESHOLD = 0.5
VIDEO_RATIO_THRESHOLD = 0.30 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. PONDASI
# ==========================================
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        ext = image_path.split('.')[-1].lower()
        if ext in ['jpg', 'jpeg']: mime_type = 'image/jpeg'
        elif ext == 'png': mime_type = 'image/png'
        else: mime_type = 'image/unknown'
        return f"data:{mime_type};base64,{encoded}"
    except Exception:
        return None 

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

# ==========================================
# 3. PANGGIL MODEL
# ==========================================
def load_mtcnn_detector():
    if 'mtcnn_detector' not in st.session_state:
        try:
            st.session_state.mtcnn_detector = MTCNN()
        except Exception as e:
            st.error(f"Error MTCNN: {e}")
            st.session_state.mtcnn_detector = None
    return st.session_state.mtcnn_detector

@st.cache_resource
def load_pytorch_model(model_path):
    try:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, 1)
        )
        
        if not os.path.exists(model_path):
            st.error(f"Model tidak ditemukan di path: {model_path}")
            return None
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error PyTorch: {e}")
        return None

# ==========================================
# 4. PREPROCESSING & LOGIKA DETEKSI
# ==========================================
data_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_indicator_frames(all_frame_data, max_frames):
    if not all_frame_data: return []
    all_frame_data.sort(key=lambda x: x[0])
    
    num_frames = len(all_frame_data)
    if num_frames <= max_frames: return all_frame_data
    
    half_max = math.ceil(max_frames / 2)
    indicator_frames = all_frame_data[:half_max] 
    indicator_frames.extend(all_frame_data[-half_max:])
    indicator_frames.sort(key=lambda x: x[0])
    return indicator_frames

def create_donut_chart(score, stroke_class):
    percentage = int(score * 100)
    svg_html = f"""<div style="width: 100px; height: 100px;"><svg viewBox="0 0 36 36" class="circular-chart"><path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" /><path class="circle {stroke_class}" stroke-dasharray="{percentage}, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" /><text x="18" y="20.35" class="percentage">{percentage}%</text></svg></div>"""
    return svg_html

# --- FUNGSI UTAMA ---
def predict_video(cap, detector, model):
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        return "Error", "Video kosong.", [], 0

    start_time = time.time() # Mulai Hitung Waktu

    indices = np.linspace(0, frame_count - 1, FRAMES_PER_VIDEO, dtype=int)
    all_scores = []
    all_frame_data = [] 
    
    progress_bar = st.progress(0)
    
    for idx_step, i in enumerate(indices):
        progress_bar.progress((idx_step + 1) / len(indices))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            faces = detector.detect_faces(frame_rgb)
        except:
            continue
        
        if not faces: continue
        if len(faces) > FACE_LIMIT:
            elapsed = time.time() - start_time
            return "TooManyFaces", float(len(faces)), [], elapsed
            
        for face in faces:
            x, y, w, h = face['box']
            if face['confidence'] < 0.85: continue
            if w < 20 or h < 20: continue
            aspect_ratio = w / h
            if aspect_ratio > 1.0 or aspect_ratio < 0.65:continue

            x1 = max(0, x - FACE_PADDING)
            y1 = max(0, y - FACE_PADDING)
            x2 = min(frame_rgb.shape[1], x + w + FACE_PADDING)
            y2 = min(frame_rgb.shape[0], y + h + FACE_PADDING)
            
            if (x2 - x1) < 10 or (y2 - y1) < 10: continue 

            face_img = frame_rgb[y1:y2, x1:x2]
            
            face_pil = Image.fromarray(face_img)
            face_tensor = data_transforms(face_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(face_tensor)
                score = torch.sigmoid(output).item()
                
            all_scores.append(score)
            
            indicator_img = cv2.resize(frame, (200, int(frame.shape[0] * (200/frame.shape[1]))))
            indicator_img_rgb = cv2.cvtColor(indicator_img, cv2.COLOR_BGR2RGB)
            all_frame_data.append((score, indicator_img_rgb))

    end_time = time.time()
    inference_time = end_time - start_time
    progress_bar.empty()

    if not all_scores:
        return "NoFace", 0.0, [], inference_time

    all_scores = np.array(all_scores)
    
    suspicious_frames_mask = all_scores > FRAME_FAKE_THRESHOLD
    suspicious_count = np.sum(suspicious_frames_mask)
    suspicious_ratio = suspicious_count / len(all_scores)
    
    if suspicious_ratio > VIDEO_RATIO_THRESHOLD:
        label = "Deepfake"
        final_score = np.mean(all_scores[suspicious_frames_mask])
    else:
        label = "Asli"
        safe_scores = all_scores[~suspicious_frames_mask]
        if len(safe_scores) > 0:
            final_score = 1.0 - np.mean(safe_scores)
        else:
            final_score = 0.0

    processed_frames = get_indicator_frames(all_frame_data, MAX_FRAMES_TO_DISPLAY)
    
    return label, final_score, processed_frames, inference_time

def process_video_file(tfile, detector, model):
    try:
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened(): return "Error", "Tidak bisa buka video", [], 0
        return predict_video(cap, detector, model)
    except Exception as e: return "Error", str(e), [], 0
    finally:
        if 'cap' in locals() and cap.isOpened(): cap.release()

# ==========================================
# 5. HALAMAN DASHBOARD
# ==========================================
def show_dashboard():
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown('<div style="margin-top: 40px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Deepfake Efficient Detector</div>', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-slogan">When you can’t trust yours<br>Then you can trust Us</div>', unsafe_allow_html=True)
        
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            st.button("How to Use ?", use_container_width=True)
        with btn_col2:
            if st.button("Try Now ->", type="primary", use_container_width=True):
                st.session_state['page'] = 'main_system'
                st.rerun()

    with col2:
        dashboard_img_base64 = get_image_base64(LOCAL_IMAGE)
        if dashboard_img_base64:
            st.markdown(f'<img src="{dashboard_img_base64}" style="width:90%; border-radius:20px; box-shadow: 0 0 30px rgba(0,212,255,0.3);">', unsafe_allow_html=True)

    st.markdown("""<div class="instruction-card">
                        <div class="instruction-header">
                        <span class="instruction-title">How to use? - Petunjuk Penggunaan :</span>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 30px;">
                        <div class="step-item">
                        <div class="step-icon">👆</div>
                        <div class="step-text">Untuk memulai, silahkan klik “Try Now” <br>untuk mencoba sistem</div>
                        </div>
                        <div class="step-item">
                        <div class="step-icon">🔎</div>
                        <div class="step-text">Klik “Browse files” untuk memilih video (MP4, FLV, MOV)</div>
                        </div>
                        <div class="step-item">
                        <div class="step-icon">🤔</div>
                        <div class="step-text">Setelah video terunggah dan tampil,<br>tekan tombol “Analisis Video Ini”</div>
                        </div>
                        <div class="step-item">
                        <div class="step-icon">▶️</div>
                        <div class="step-text">Sistem akan memproses video, mendeteksi wajah,<br>dan menganalisisnya.</div>
                        </div>
                        <div class="step-item">
                        <div class="step-icon">🕵️‍♂️</div>
                        <div class="step-text">Hasil klasifikasi akan muncul dengan output<br>Asli atau Deepfake beserta tingkat akurasi</div>
                        </div>
                        <div class="step-item">
                        <div class="step-icon">👨🏻‍💻</div>
                        <div class="step-text">Frame yang menjadi indikator utama akan<br>ditampilkan di bagian paling bawah.</div>
                        </div>
                        </div>
                        </div>""", unsafe_allow_html=True)

# ==========================================
# 6. HALAMAN UTAMA (MAIN SYSTEM)
# ==========================================
def show_main_system():
    with st.sidebar:
        if st.button("⬅️ Kembali ke Dashboard"):
            st.session_state['page'] = 'dashboard'
            st.rerun()
        st.header("Selamat datang di DEFEND (Deepfake Efficient Detector)")
        st.header("📌 Petunjuk Penggunaan :")
        st.markdown("""
                    Deepfake Video based Detection with Transfer Learning EfficientNetB0 Model
                    1.  Klik tombol "Browse files" untuk memilih video dari perangkat komputer Anda.
                    2.  Setelah video terunggah, klik tombol "Analisis Video Ini" dibawah display.
                    3.  Mohon tunggu sampai proses analisis selesai.
                    4.  Hasil klasifikasi akan muncul di bawah.
                    Catatan: Format Video (.mp4), Pastikan ukuran video tidak melebihi 200MB.""")

    robot_icon_base64 = get_image_base64(LOCAL_ROBOT_ICON)
    img_src = robot_icon_base64 if robot_icon_base64 else ""
    st.markdown(f"""
        <div class="header-container">
            <img src="{img_src}" class="header-icon">
            <div class="header-text-block"><h1 class="main-title">DEFEND</h1></div>
        </div>
        <hr class="custom-divider">
        <p class="main-description">Deepfake Efficient Detector</p>
    """, unsafe_allow_html=True)

    detector = load_mtcnn_detector()
    model = load_pytorch_model(MODEL_PATH)
    
    if detector is None or model is None:
        st.error("Gagal memuat model. Cek path file.")
        return

    uploaded_file = st.file_uploader("", type=["mp4", "flv", "mov"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        st.divider() 
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("Analisis Video Ini", use_container_width=True)
        
        if analyze_button:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                
            with st.spinner("Sedang menganalisis..."):
                label, score, frames, duration = process_video_file(tfile, detector, model)
            
            os.remove(tfile.name)

            # --- OUTPUT ---
            if label == "Error":
                st.markdown(f"""<div class="result-container"><div class="result-row bg-blue"><span class="label-col">Error</span><span class="value-col">: {score}</span></div></div>""", unsafe_allow_html=True)
                
            elif label == "NoFace":
                st.markdown(f"""<div class="result-container">
                <div class="result-row bg-black">
                <span class="label-col">Hasil</span><span class="value-col">: Tidak ada wajah terdeteksi</span>
                </div>
                <div class="result-row bg-blue">
                <span class="label-col">Sistem</span><span class="value-col">: Analisis Selesai</span>
                </div>
                </div>""", unsafe_allow_html=True)

            elif label == "TooManyFaces":
                detected_faces = int(score) 
                st.markdown(f"""<div class="result-container">
                <div class="result-row bg-orange">
                <span class="label-col">Hasil</span><span class="value-col">: Terlalu banyak wajah ({detected_faces} Wajah)</span>
                </div>
                <div class="result-row bg-blue">
                <span class="label-col">Sistem</span><span class="value-col">: {detected_faces} Wajah Terdeteksi</span>
                </div>
                <div class="notification-text"><b>Notifikasi :</b><br>Sistem menemukan <b>{detected_faces} wajah</b>. Model ini tidak dirancang untuk memproses lebih dari {FACE_LIMIT} wajah sekaligus.</div>
                </div>""", unsafe_allow_html=True)

            else:
                if label == "Deepfake":
                    res_class = "bg-red"
                    stroke_class = "stroke-red"
                    res_text = ": Terdeteksi DEEPFAKE"
                else: 
                    res_class = "bg-green"
                    stroke_class = "stroke-green"
                    res_text = ": Terdeteksi ASLI"
                
                donut_html = create_donut_chart(score, stroke_class)

                st.markdown(f"""<div class="result-container">
                                    <div class="result-row {res_class}">
                                    <span class="label-col">Hasil</span><span class="value-col">{res_text}</span>
                                    </div>
                                    <div class="result-row bg-blue">
                                    <span class="label-col">Waktu</span><span class="value-col">: {duration:.2f} Detik</span>
                                    </div>
                                    <div class="confidence-section">
                                    <div class="confidence-text">
                                    <div class="confidence-title">Tingkat Akurasi</div>
                                    <div class="confidence-value">{score*100:.0f}%</div>
                                    </div>
                                    {donut_html}
                                    </div>
                                    </div>""", unsafe_allow_html=True)

                if frames: 
                    st.divider()
                    st.subheader("📸 Frame Indikator")
                    cols = st.columns(5) 
                    for i, (f_score, f_img) in enumerate(frames):
                        is_fake = f_score > FRAME_FAKE_THRESHOLD
                        f_lbl = "Deepfake" if is_fake else "Asli"
                        with cols[i % 5]:
                            st.image(f_img, caption=f"#{i+1} {f_lbl} ({f_score*100:.0f}%)", use_container_width=True)

# ==========================================
# 7. ROUTER UTAMA
# ==========================================
def main():
    st.set_page_config(page_title="DEFEND", page_icon="🤖", layout="wide")
    load_css("style.css")

    if 'page' not in st.session_state:
        st.session_state['page'] = 'dashboard'

    if st.session_state['page'] == 'dashboard':
        show_dashboard()
    elif st.session_state['page'] == 'main_system':
        show_main_system()

if __name__ == "__main__":
    main()