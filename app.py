import streamlit as st
import os
import cv2
import whisper
import shutil
import base64
from scenedetect import detect, ContentDetector

# ===============================
# 設定
# ===============================
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "temp_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# ユーティリティ
# ===============================
def format_time(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02}:{s:02}"


def clear_output():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# Whisper モデル読み込み（キャッシュ）
# ===============================
@st.cache_resource
def load_whisper():
    return whisper.load_model("medium")  # ここで精度UP（small → medium）


# ===============================
# シーン抽出（新方式）
# ===============================
def extract_scenes(video_path):
    scenes = detect(video_path, ContentDetector(threshold=27.0))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps

    scene_data = []

    # 最初のシーンが0秒で始まらない場合は追加
    if len(scenes) == 0 or scenes[0][0].get_seconds() > 1.0:
        scene_data.append({
            "start": 0.0,
            "end": scenes[0][0].get_seconds() if scenes else duration,
            "img": None,
            "text": "",
        })

    # SceneDetect 結果
    for start_time, end_time in scenes:
        scene_data.append({
            "start": start_time.get_seconds(),
            "end": end_time.get_seconds(),
            "img": None,
            "text": "",
        })

    # スクショ
    for i, sc in enumerate(scene_data):
        capture_time = sc["start"] + 0.5 if (sc["end"] - sc["start"]) > 1 else sc["start"]
        cap.set(cv2.CAP_PROP_POS_MSEC, capture_time * 1000)
        ret, frame = cap.read()

        if ret:
            img_path = os.path.join(OUTPUT_DIR, f"scene_{i:03d}.jpg")
            cv2.imwrite(img_path, frame)
            sc["img"] = img_path

    cap.release()
    return scene_data


# ===============================
# 書き起こし
# ===============================
def transcribe(video_path):
    model = load_whisper()
    with st.spinner("Whisper が音声を解析中…"):
        result = model.transcribe(video_path, language="ja")
    return result["segments"]


# ===============================
# シーンと字幕の結合
# ===============================
def align(scenes, segments):
    for seg in segments:
        mid = (seg["start"] + seg["end"]) / 2
        text = seg["text"]

        matched = False
        for sc in scenes:
            if sc["start"] <= mid < sc["end"]:
                sc["text"] += text + " "
                matched = True
                break

        if not matched:
            scenes[-1]["text"] += text + " "

    return scenes


# ===============================
# HTML 生成（横スクロール UI）
# ===============================
def render_scenes(scenes):
    html = """
    <style>
    .scroll-container {
        white-space: nowrap;
        overflow-x: auto;
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 10px;
    }
    .scene-card {
        display: inline-block;
        width: 250px;
        margin-right: 15px;
        vertical-align: top;
        border-radius: 10px;
        background: #fafafa;
        padding: 10px;
        border: 1px solid #ddd;
    }
    .scene-img {
        width: 100%;
        border-radius: 8px;
    }
    .scene-text {
        font-size: 13px;
        margin-top: 8px;
        white-space: normal;
    }
    </style>
    <div class="scroll-container">
    """

    for sc in scenes:
        if sc["img"] and os.path.exists(sc["img"]):
            with open(sc["img"], "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
        else:
            img_b64 = ""

        html += f"""
        <div class="scene-card">
            <img src="data:image/jpeg;base64,{img_b64}" class="scene-img"/>
            <div class="scene-text">{sc['text']}</div>
        </div>
        """

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ===============================
# メイン UI
# ===============================
st.title("🎬 動画 → シーン解析 & 書き起こし（Whisper ローカル版）")

uploaded = st.file_uploader("動画ファイルをアップロード", type=["mp4", "mov", "avi"])

if uploaded:
    video_path = os.path.join(UPLOAD_DIR, uploaded.name)
    with open(video_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.success("アップロード完了！")

    if st.button("🚀 解析スタート"):
        clear_output()

        scenes = extract_scenes(video_path)
        segments = transcribe(video_path)
        scenes = align(scenes, segments)

        st.subheader("📸 シーン & テキスト（横スクロール）")
        render_scenes(scenes)

        st.success("解析が完了しました！")
