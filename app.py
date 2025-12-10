import streamlit as st
import os
import cv2
import whisper
import shutil
import datetime
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import pandas as pd

# --- 設定 ---
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "temp_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 関数: 時間表示 ---
def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

# --- 関数: フォルダリセット ---
def clear_output_folder():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 関数: シーン抽出 ---
def extract_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    
    cap = cv2.VideoCapture(video_path)
    scenes_data = []

    progress_bar = st.progress(0, text="シーン検出中...")
    total_scenes = len(scene_list)

    # 最初のシーン(開始0秒)を必ず追加
    if total_scenes > 0 and scene_list[0][0].get_seconds() > 0:
        start_time = 0.0
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        ret, frame = cap.read()
        if ret:
            img_filename = "scene_000_0s.jpg"
            img_path = os.path.join(OUTPUT_DIR, img_filename)
            cv2.imwrite(img_path, frame)
            scenes_data.append({
                "time_str": format_time(start_time),
                "seconds": start_time,
                "img_path": img_path
            })

    # 通常シーン
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        ret, frame = cap.read()
        
        if ret:
            img_filename = f"scene_{i:03d}_{int(start_time)}s.jpg"
            img_path = os.path.join(OUTPUT_DIR, img_filename)
            cv2.imwrite(img_path, frame)
            
            scenes_data.append({
                "time_str": format_time(start_time),
                "seconds": start_time,
                "img_path": img_path
            })
        
        if total_scenes > 0:
            progress_bar.progress(min((i + 1) / total_scenes, 1.0))

    cap.release()
    progress_bar.empty()
    return scenes_data

# --- Whisperモデル読み込み ---
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

def transcribe_audio(video_path):
    model = load_whisper_model()
    with st.spinner("音声を解析中..."):
        result = model.transcribe(video_path, language="ja")
    return result["segments"]

# --- シーンとテキスト結合 ---
def align_scenes_and_text(scenes, segments):
    aligned_data = []
    
    for i, scene in enumerate(scenes):
        scene_start = scene["seconds"]
        next_scene_start = scenes[i+1]["seconds"] if i+1 < len(scenes) else float('inf')
        
        matched_texts = [
            seg["text"] for seg in segments
            if scene_start <= seg["start"] < next_scene_start
        ]
        
        combined_text = "\n".join(matched_texts)
        
        aligned_data.append({
            "time": scene["time_str"],
            "image": scene["img_path"],
            "text": combined_text
        })
    return aligned_data


# ==========================================
# UI
# ==========================================
st.set_page_config(page_title="動画解析アプリ Pro", layout="wide")

st.title("🎥 動画解析アプリ Pro")
st.markdown("Gemini版と同じ挙動で動くように最適化済み。")

# MIME制限を外して動画アップロード可能にする
uploaded_file = st.file_uploader("動画ファイルをアップロード", accept_multiple_files=False)

if uploaded_file is not None:
    video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"読み込み完了: {uploaded_file.name}")

    if st.button("🚀 解析スタート"):
        clear_output_folder()

        scenes = extract_scenes(video_path)
        segments = transcribe_audio(video_path)
        aligned_data = align_scenes_and_text(scenes, segments)

        st.divider()
        st.subheader("📊 解析結果（スプレッドシート貼り付け用）")

        num_scenes = len(aligned_data)

        # 時間
        cols_time = st.columns(num_scenes)
        for i, col in enumerate(cols_time):
            col.write(f"**{aligned_data[i]['time']}**")

        # 画像
        cols_img = st.columns(num_scenes)
        for i, col in enumerate(cols_img):
            col.image(aligned_data[i]["image"], use_column_width=True)

        # テキスト
        cols_text = st.columns(num_scenes)
        for i, col in enumerate(cols_text):
            col.text_area("", aligned_data[i]["text"], height=150, key=f"t_{i}")

        # CSVダウンロード
        df = pd.DataFrame(aligned_data)
        csv = df.to_csv(index=False).encode("utf-8_sig")
        st.download_button("📥 CSVでダウンロード", csv, "video_analysis.csv")
