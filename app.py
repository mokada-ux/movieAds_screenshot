import streamlit as st
import os
import cv2
import whisper
import shutil
import zipfile
import datetime
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# --- 設定 ---
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "temp_outputs"
# フォルダがなければ作成
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
    # threshold=27.0 は感度の標準値。動きが少ない動画なら下げてください。
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    
    cap = cv2.VideoCapture(video_path)
    scenes_data = []

    progress_bar = st.progress(0, text="シーン検出中...")
    total_scenes = len(scene_list)
    
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
                "img_path": img_path,
                "filename": img_filename
            })
        
        if total_scenes > 0:
            progress_bar.progress(min((i + 1) / total_scenes, 1.0))

    cap.release()
    progress_bar.empty()
    return scenes_data

# --- 関数: 音声書き起こし ---
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base") # 精度重視なら "small" や "medium" に変更

def transcribe_audio(video_path):
    model = load_whisper_model()
    # st.spinner で処理中を表示
    with st.spinner("AIが音声を解析しています... (動画の長さにより数分かかります)"):
        result = model.transcribe(video_path)
    return result["segments"]

# --- 関数: ZIP作成 ---
def create_zip(file_paths):
    zip_path = os.path.join(OUTPUT_DIR, "scenes.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in file_paths:
            zipf.write(file, os.path.basename(file))
    return zip_path

# ==========================================
# メインUI (Streamlit)
# ==========================================
st.set_page_config(page_title="動画解析アプリ", layout="wide")

st.title("🎥 動画シーン & 字幕抽出ツール")
st.markdown("動画をアップするだけで「**場面写真**」と「**文字起こし**」を一括生成します。")

# サイドバー設定
with st.sidebar:
    st.header("設定")
    enable_scene = st.checkbox("シーン画像を抽出する", value=True)
    enable_text = st.checkbox("音声を文字起こしする", value=True)
    st.divider()
    st.info("※ FFmpegがインストールされている必要があります。")

uploaded_file = st.file_uploader("動画ファイルをアップロード", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    # ファイル保存
    video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"読み込み完了: {uploaded_file.name}")

    if st.button("🚀 解析スタート", type="primary"):
        clear_output_folder()
        
        # 1. シーン抽出
        scenes = []
        if enable_scene:
            st.subheader("📸 検出されたシーン")
            scenes = extract_scenes(video_path)
            
            if scenes:
                # ギャラリー表示
                cols = st.columns(4)
                img_paths = []
                for i, scene in enumerate(scenes):
                    with cols[i % 4]:
                        st.image(scene["img_path"], caption=scene["time_str"])
                    img_paths.append(scene["img_path"])
                
                # ZIPダウンロードボタン
                zip_path = create_zip(img_paths)
                with open(zip_path, "rb") as fp:
                    st.download_button(
                        label="📥 全画像をZIPでダウンロード",
                        data=fp,
                        file_name="scene_images.zip",
                        mime="application/zip"
                    )
            else:
                st.warning("シーンの変化が検出されませんでした。")
        
        st.divider()

        # 2. 文字起こし
        if enable_text:
            st.subheader("📝 文字起こし結果")
            segments = transcribe_audio(video_path)
            
            full_text = ""
            for segment in segments:
                line = f"[{format_time(segment['start'])}] {segment['text']}\n"
                full_text += line
            
            # テキストエリア表示
            st.text_area("書き起こし内容", full_text, height=300)
            
            # テキストダウンロードボタン
            st.download_button(
                label="📥 テキストファイル(.txt)で保存",
                data=full_text,
                file_name="transcription.txt",
                mime="text/plain"
            )