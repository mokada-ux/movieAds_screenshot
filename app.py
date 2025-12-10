import streamlit as st
import os
import cv2
import whisper
import shutil
import base64
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# ====== 設定 ====== #
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "temp_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="動画解析 Pro", layout="wide")

# ====== CSS カスタム ====== #
st.markdown("""
<style>
/* 全体フォント */
html, body, [class*="css"]  {
    font-family: "Inter", "Noto Sans JP", sans-serif;
}

/* 横スクロールコンテナ */
.scene-container {
    display: flex;
    flex-direction: row;
    overflow-x: auto;
    gap: 20px;
    padding-bottom: 20px;
    white-space: nowrap;
}

/* 1シーンのカードデザイン */
.scene-card {
    display: inline-block;
    width: 280px;
    background: #ffffff10;
    backdrop-filter: blur(6px);
    padding: 12px;
    border-radius: 14px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.15);
}

/* サムネイル画像 */
.scene-img {
    width: 100%;
    border-radius: 10px;
    margin-bottom: 8px;
    border: 1px solid #ddd;
}

/* テキスト領域 */
.scene-text {
    font-size: 14px;
    line-height: 1.5;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)


# ====== 関数類 ====== #

def clear_output_folder():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def format_time(seconds):
    seconds = int(seconds)
    m = seconds // 60
    s = seconds % 60
    return f"{m:02}:{s:02}"


def extract_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps

    scenes = []
    if not scene_list or scene_list[0][0].get_seconds() > 1.0:
        scenes.append({"start": 0.0, "end": scene_list[0][0].get_seconds() if scene_list else duration})

    for s in scene_list:
        scenes.append({"start": s[0].get_seconds(), "end": s[1].get_seconds()})

    # 画像保存
    for i, sc in enumerate(scenes):
        cap.set(cv2.CAP_PROP_POS_MSEC, int((sc["start"] + 0.3) * 1000))
        ret, frame = cap.read()
        if ret:
            filename = f"scene_{i:03}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frame)
            sc["img"] = os.path.join(OUTPUT_DIR, filename)

        sc["time_str"] = format_time(sc["start"])

    cap.release()
    return scenes


@st.cache_resource
def load_whisper():
    return whisper.load_model("small")   # ←精度UP


def transcribe_audio(path):
    model = load_whisper()
    result = model.transcribe(path, language="ja")
    return result["segments"]


def align(scenes, segments):
    for sc in scenes:
        sc["text"] = ""

    for seg in segments:
        mid = (seg["start"] + seg["end"]) / 2
        for sc in scenes:
            if sc["start"] <= mid < sc["end"]:
                sc["text"] += seg["text"] + "\n"
                break
        else:
            scenes[-1]["text"] += seg["text"] + "\n"

    return scenes


# ====== UI ====== #

st.title("🎥 動画解析 Pro（UI強化版）")
uploaded = st.file_uploader("動画ファイルをアップロード", type=["mp4", "mov", "avi"])

if uploaded is not None:
    video_path = os.path.join(UPLOAD_DIR, uploaded.name)
    with open(video_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.success("動画アップロード完了！")

    if st.button("🚀 解析スタート", type="primary"):
        clear_output_folder()

        with st.spinner("シーン抽出中..."):
            scenes = extract_scenes(video_path)

        with st.spinner("音声解析中...（Whisper small）"):
            segments = transcribe_audio(video_path)

        scenes = align(scenes, segments)

        st.subheader("🎬 シーン一覧（横スクロール可）")

        # ===== 横スクロール HTML生成 ===== #
        html = """<div class="scene-container">"""

        for sc in scenes:
            with open(sc["img"], "rb") as f:
                encoded = base64.b64encode(f.read()).decode()

            html += f"""
            <div class="scene-card">
                <img src="data:image/jpeg;base64,{encoded}" class="scene-img" />
                <div><b>⏱ {sc['time_str']}〜</b></div>
                <div class="scene-text">{sc['text']}</div>
            </div>
            """

        html += "</div>"

        st.markdown(html, unsafe_allow_html=True)

        st.subheader("📊 スプレッドシート用（横並び）")

        tsv = "\t".join([s["text"].replace("\n", " ") for s in scenes])
        st.code(tsv, language="text")
