import streamlit as st
import tempfile
import os
import whisper
import cv2
from PIL import Image
import numpy as np

# Whisperモデルのロード（smallで安定）
@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

st.title("動画シーン × テキスト抽出ツール")

uploaded_file = st.file_uploader("動画ファイルをアップロード", type=["mp4", "mov", "mkv"])

if uploaded_file is not None:

    # 一時保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(uploaded_file)

    st.write("### ① シーン検出中...")

    # --- シーン検出 ---
    def detect_scenes(video_path, threshold=30, min_scene_len=1):
        cap = cv2.VideoCapture(video_path)
        scenes = []
        last_frame = None
        start_time = 0

        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if last_frame is not None:
                diff = cv2.absdiff(gray, last_frame)
                score = diff.mean()

                if score > threshold:
                    end_time = frame_count / fps
                    if end_time - start_time >= min_scene_len:
                        scenes.append((start_time, end_time))
                    start_time = end_time

            last_frame = gray
            frame_count += 1

        cap.release()
        return scenes

    scenes = detect_scenes(video_path)
    st.success(f"{len(scenes)} 個のシーンを検出しました")

    st.write("### ② Whisperで音声→テキスト変換中...（精度強化）")

    # Whisper 高精度設定
    result = model.transcribe(
        audio=video_path,
        verbose=True,
        temperature=0.0,
        condition_on_previous_text=True,
        fp16=False
    )

    segments = result["segments"]

    # --- セグメントをシーンごとにマッピング ---
    def match_segments_to_scenes(scenes, segments):
        scene_texts = []
        for (start, end) in scenes:
            text = ""
            for seg in segments:
                if seg["start"] >= start and seg["start"] <= end:
                    text += seg["text"] + " "
            scene_texts.append(text.strip())
        return scene_texts

    scene_texts = match_segments_to_scenes(scenes, segments)

    st.write("### ③ シーンごとの画像＆テキスト生成中...")

    # スクショ作成
    def capture_frame_at(video_path, time_sec):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    screenshots = []
    for (start, end) in scenes:
        mid = (start + end) / 2
        img = capture_frame_at(video_path, mid)
        screenshots.append(img)

    # ---- 横スクロール表示 ----
    st.write("### ④ シーン一覧（横スクロール）")

    # CSS で横スクロールコンテナを作成
    st.markdown("""
    <style>
    .scroll-container {
        display: flex;
        overflow-x: auto;
        gap: 20px;
        padding-bottom: 20px;
        white-space: nowrap;
    }
    .scene-item {
        display: inline-block;
        text-align: center;
        width: 300px;
    }
    .scene-img {
        width: 100%;
        border-radius: 8px;
    }
    .scene-text {
        font-size: 14px;
        margin-top: 8px;
        background: #f0f0f0;
        padding: 8px;
        border-radius: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

    html = '<div class="scroll-container">'

    for img, text in zip(screenshots, scene_texts):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            img.save(tmp_img.name)
            img_path = tmp_img.name

        html += f"""
        <div class="scene-item">
            <img src="data:image/jpeg;base64,{base64.b64encode(open(img_path,'rb').read()).decode()}" class="scene-img"/>
            <div class="scene-text">{text}</div>
        </div>
        """

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)

