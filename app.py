import streamlit as st
import os
import shutil
import datetime
import subprocess
import whisper
import pandas as pd
from scenedetect import detect, ContentDetector

# ===============================
# 設定
# ===============================
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "temp_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# 関数類
# ===============================

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


def clear_output_folder():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_scenes_ffmpeg_safe(video_path):
    """
    SceneDetect でシーンのみ検出し、
    画像は FFmpeg で確実に出力する安全版。
    """
    st.info("シーン検出中...")

    # SceneDetect でシーン抽出（時間だけ取得）
    scene_list = detect(video_path, ContentDetector())

    scenes_data = []

    for i, scene in enumerate(scene_list):
        start_sec = scene[0].get_seconds()
        img_path = os.path.join(OUTPUT_DIR, f"{i:03d}.jpg")

        # FFmpeg で指定時間の1フレームを抽出
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_sec),
            "-i", video_path,
            "-vframes", "1",
            img_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        scenes_data.append({
            "time_str": format_time(start_sec),
            "seconds": start_sec,
            "img_path": img_path
        })

    return scenes_data


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")


def transcribe_audio(video_path):
    model = load_whisper_model()
    with st.spinner("音声を解析中..."):
        result = model.transcribe(video_path, language="ja")
    return result["segments"]


def align_scenes_and_text(scenes, segments):
    aligned = []

    for i, scene in enumerate(scenes):
        scene_start = scene["seconds"]
        next_scene_start = scenes[i+1]["seconds"] if i+1 < len(scenes) else float('inf')

        matched_texts = [
            seg["text"]
            for seg in segments
            if scene_start <= seg["start"] < next_scene_start
        ]

        aligned.append({
            "time": scene["time_str"],
            "image": scene["img_path"],
            "text": "\n".join(matched_texts)
        })

    return aligned


# ===============================
# UI
# ===============================
st.set_page_config(page_title="動画解析アプリ Pro", layout="wide")

st.title("🎥 動画解析アプリ Pro")
st.markdown("Whisper-small + SceneDetect + FFmpeg の安定動作版。")


uploaded_file = st.file_uploader("動画ファイルをアップロード", type=["mp4", "mov", "m4v", "avi", "webm"])

if uploaded_file is not None:
    video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"読み込み完了: {uploaded_file.name}")

    if st.button("🚀 解析スタート"):
        clear_output_folder()

        # シーン静止画（FFmpeg）
        scenes = extract_scenes_ffmpeg_safe(video_path)

        # Whisper
        segments = transcribe_audio(video_path)

        # 結合
        aligned_data = align_scenes_and_text(scenes, segments)

        st.divider()
        st.subheader("📊 解析結果（スプレッドシート貼り付け用）")

        num = len(aligned_data)

        # 時間
        cols_time = st.columns(num)
        for i, col in enumerate(cols_time):
            col.write(f"**{aligned_data[i]['time']}**")

        # 画像
        cols_img = st.columns(num)
        for i, col in enumerate(cols_img):
            col.image(aligned_data[i]["image"], use_column_width=True)

        # テキスト
        cols_text = st.columns(num)
        for i, col in enumerate(cols_text):
            col.text_area("", aligned_data[i]["text"], height=150, key=f"t_{i}")

        # CSV
        df = pd.DataFrame(aligned_data)
        csv = df.to_csv(index=False).encode("utf-8_sig")
        st.download_button("📥 CSVダウンロード", csv, "video_analysis.csv")
