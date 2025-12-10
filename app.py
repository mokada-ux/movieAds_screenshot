import streamlit as st
import os
import cv2
import whisper
import shutil
import datetime
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# ===========================
# 初期設定
# ===========================
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "temp_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# Utility Functions
# ===========================
def format_time(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    rem_seconds = seconds % 60
    return f"{minutes:02}:{rem_seconds:02}"

def clear_output_folder():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# シーン抽出
# ===========================
def extract_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("動画の読み込みに失敗しました。")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0

    scenes = []

    # 最初のシーンを強制追加
    if not scene_list or scene_list[0][0].get_seconds() > 1.0:
        scenes.append({
            "start": 0.0,
            "end": scene_list[0][0].get_seconds() if scene_list else duration,
            "time_str": format_time(0),
            "img_path": None
        })

    # SceneDetect の結果を整形
    for i, scene in enumerate(scene_list):
        start = scene[0].get_seconds()
        end = scene[1].get_seconds()
        scenes.append({
            "start": start,
            "end": end,
            "time_str": format_time(start),
            "img_path": None
        })

    # -------------------------
    # シーンごとの画像を保存
    # -------------------------
    progress = st.progress(0, text="シーン画像抽出中...")
    total = len(scenes)

    for i, scene in enumerate(scenes):
        scene_len = scene["end"] - scene["start"]
        capture_point = scene["start"] + (0.5 if scene_len > 1.0 else 0.0)

        cap.set(cv2.CAP_PROP_POS_MSEC, capture_point * 1000)
        ret, frame = cap.read()

        if ret:
            img_path = os.path.join(OUTPUT_DIR, f"scene_{i:03d}.jpg")
            cv2.imwrite(img_path, frame)
            scene["img_path"] = img_path
        else:
            scene["img_path"] = None

        progress.progress((i + 1) / total)

    progress.empty()
    cap.release()
    return scenes

# ===========================
# Whisper 音声書き起こし
# ===========================
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def transcribe_audio(video_path):
    model = load_whisper_model()
    with st.spinner("AI が音声を解析中..."):
        result = model.transcribe(video_path, language="ja")
    return result["segments"]

# ===========================
# シーン×音声アライメント
# ===========================
def align_scenes_and_text(scenes, segments):
    for scene in scenes:
        scene["text_list"] = []

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_mid = (seg_start + seg_end) / 2

        matched = False

        for scene in scenes:
            if scene["start"] <= seg_mid < scene["end"]:
                scene["text_list"].append(seg["text"])
                matched = True
                break

        if not matched and scenes:
            scenes[-1]["text_list"].append(seg["text"])

    # 結合
    for scene in scenes:
        scene["final_text"] = "\n".join(scene["text_list"])

    return scenes

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="動画解析アプリ Pro", layout="wide")
st.title("🎥 動画解析 & スプレッドシート貼り付けツール（ローカルWhisper版）")

uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi"])

if uploaded_file:
    video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"アップロード完了: {uploaded_file.name}")

    if st.button("🚀 解析スタート", type="primary"):
        clear_output_folder()

        # Step1: シーン抽出
        scenes = extract_scenes(video_path)

        # Step2: 音声書き起こし
        segments = transcribe_audio(video_path)

        # Step3: アライメント
        aligned = align_scenes_and_text(scenes, segments)

        st.divider()
        st.subheader("1. 解析結果プレビュー")

        cols = st.columns(3)
        for i, item in enumerate(aligned):
            with cols[i % 3]:
                if item["img_path"]:
                    st.image(item["img_path"], use_column_width=True)
                st.caption(f"シーン {i+1}（{item['time_str']}〜）")
                st.text_area("内容", item["final_text"], height=110, key=f"text_{i}")

        st.divider()
        st.subheader("2. スプレッドシート貼り付け用")

        tsv_text = "\t".join([item["final_text"].replace("\n", " ") for item in aligned])
        st.code(tsv_text, language="text")
