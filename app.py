import streamlit as st
import os
import shutil
import datetime
import tempfile
import whisper
import pandas as pd
from scenedetect import detect, ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg

# ===============================
# 設定
# ===============================
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "temp_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# 関数
# ===============================

def format_time(seconds):
    """秒 → 00:00:00 形式へ"""
    return str(datetime.timedelta(seconds=int(seconds)))


def clear_output_folder():
    """出力フォルダ初期化"""
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_scenes_ffmpeg(video_path):
    """SceneDetect + FFmpeg でシーン静止画を抽出"""
    st.info("シーン検出中...")

    # SceneDetect でシーン抽出
    scene_list = detect(video_path, ContentDetector())

    # FFmpeg での静止画出力（jpg）
    split_video_ffmpeg(
        video_path,
        scene_list,
        output_dir=OUTPUT_DIR,
        filename_template="$SCENE_NUMBER.jpg",
        format="jpg"
    )

    # ファイル名順に並び替え
    images = sorted(os.listdir(OUTPUT_DIR))

    scenes_data = []
    for i, scene in enumerate(scene_list):
        start_sec = scene[0].get_seconds()
        img_file = images[i] if i < len(images) else None
        if img_file:
            scenes_data.append({
                "time_str": format_time(start_sec),
                "seconds": start_sec,
                "img_path": os.path.join(OUTPUT_DIR, img_file)
            })

    return scenes_data


@st.cache_resource
def load_whisper_model():
    """Whisper-small をキャッシュ読み込み"""
    return whisper.load_model("small")


def transcribe_audio(video_path):
    """Whisper で文字起こし"""
    model = load_whisper_model()
    with st.spinner("音声を解析中..."):
        result = model.transcribe(video_path, language="ja")
    return result["segments"]


def align_scenes_and_text(scenes, segments):
    """シーンとテキストを紐付け"""
    aligned_data = []

    for i, scene in enumerate(scenes):
        scene_start = scene["seconds"]
        next_scene_start = scenes[i+1]["seconds"] if i+1 < len(scenes) else float('inf')

        matched_texts = [
            seg["text"]
            for seg in segments
            if scene_start <= seg["start"] < next_scene_start
        ]

        aligned_data.append({
            "time": scene["time_str"],
            "image": scene["img_path"],
            "text": "\n".join(matched_texts)
        })

    return aligned_data


# ===============================
# UI
# ===============================
st.set_page_config(page_title="動画解析アプリ Pro", layout="wide")

st.title("🎥 動画解析アプリ Pro")
st.markdown("Whisper-small × SceneDetect(video_splitter) で最適化済み。Gemini版と同等の動作。")

uploaded_file = st.file_uploader("動画ファイルをアップロード (mp4/mov など)", type=["mp4", "mov", "m4v", "avi", "webm"])

if uploaded_file is not None:
    video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"読み込み完了: {uploaded_file.name}")

    if st.button("🚀 解析スタート"):
        clear_output_folder()

        # --- シーン静止画抽出（FFmpeg） ---
        scenes = extract_scenes_ffmpeg(video_path)

        # --- Whisper で文字起こし ---
        segments = transcribe_audio(video_path)

        # --- シーンとテキストを結合 ---
        aligned_data = align_scenes_and_text(scenes, segments)

        # -----------------------------------
        # 結果表示UI
        # -----------------------------------

        st.divider()
        st.subheader("📊 解析結果（スプレッドシート貼り付け用）")

        num_scenes = len(aligned_data)

        # ⏱ 時間
        cols_time = st.columns(num_scenes)
        for i, col in enumerate(cols_time):
            col.write(f"**{aligned_data[i]['time']}**")

        # 🖼 画像
        cols_img = st.columns(num_scenes)
        for i, col in enumerate(cols_img):
            col.image(aligned_data[i]["image"], use_column_width=True)

        # 📝 テキスト
        cols_text = st.columns(num_scenes)
        for i, col in enumerate(cols_text):
            col.text_area("", aligned_data[i]["text"], height=150, key=f"t_{i}")

        # CSVダウンロード
        df = pd.DataFrame(aligned_data)
        csv = df.to_csv(index=False).encode("utf-8_sig")
        st.download_button("📥 CSVでダウンロード", csv, "video_analysis.csv")
