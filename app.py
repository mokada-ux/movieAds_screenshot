import streamlit as st
import os
import cv2
import whisper
import shutil
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# ============================
# 設定
# ============================
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "temp_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# Utility
# ============================
def format_time(seconds):
    seconds = int(seconds)
    return f"{seconds//60:02}:{seconds%60:02}"

def clear_output_folder():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# シーン抽出
# ============================
def extract_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps

    scenes = []

    # 最初のシーンを強制追加
    if not scene_list or scene_list[0][0].get_seconds() > 1.0:
        scenes.append({
            "start": 0.0,
            "end": scene_list[0][0].get_seconds() if scene_list else duration,
            "time_str": "00:00",
            "img_path": None
        })

    # SceneDetect の結果
    for scene in scene_list:
        start = scene[0].get_seconds()
        end = scene[1].get_seconds()
        scenes.append({
            "start": start,
            "end": end,
            "time_str": format_time(start),
            "img_path": None
        })

    # --------- 画像保存 ---------
    progress = st.progress(0, text="シーン画像抽出中...")
    total = len(scenes)

    for i, scene in enumerate(scenes):
        capture_point = scene["start"] + 0.5 if scene["end"] - scene["start"] > 1.0 else scene["start"]
        cap.set(cv2.CAP_PROP_POS_MSEC, capture_point * 1000)
        ret, frame = cap.read()

        if ret:
            img_path = os.path.join(OUTPUT_DIR, f"scene_{i:03d}.jpg")
            cv2.imwrite(img_path, frame)
            scene["img_path"] = img_path

        progress.progress((i+1)/total)

    progress.empty()
    cap.release()
    return scenes

# ============================
# Whisper（精度強化版）
# ============================
@st.cache_resource
def load_whisper_model():
    # ★ 精度強化（model="medium"）
    return whisper.load_model("medium")

def transcribe_audio(video_path):
    model = load_whisper_model()
    with st.spinner("Whisper（高精度モデル）で音声解析中…"):
        result = model.transcribe(video_path, language="ja")
    return result["segments"]

# ============================
# シーン × テキスト アライメント
# ============================
def align_scenes_and_text(scenes, segments):
    for scene in scenes:
        scene["text_list"] = []

    for seg in segments:
        mid = (seg["start"] + seg["end"]) / 2
        matched = False

        for scene in scenes:
            if scene["start"] <= mid < scene["end"]:
                scene["text_list"].append(seg["text"])
                matched = True
                break

        if not matched:
            scenes[-1]["text_list"].append(seg["text"])

    for scene in scenes:
        scene["final_text"] = "\n".join(scene["text_list"])

    return scenes

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="動画解析アプリ Pro", layout="wide")
st.title("🎥 高精度動画解析 & 横スクロール表示アプリ")

uploaded = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi"])

if uploaded:
    video_path = os.path.join(UPLOAD_DIR, uploaded.name)
    with open(video_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.success(f"アップロード完了：{uploaded.name}")

    if st.button("🚀 解析スタート", type="primary"):
        clear_output_folder()

        scenes = extract_scenes(video_path)
        segments = transcribe_audio(video_path)
        aligned = align_scenes_and_text(scenes, segments)

        st.divider()
        st.subheader("1. シーン一覧（横スクロール）")

        # ============================
        # 横スクロール CSS
        # ============================
        st.markdown("""
        <style>
        .scroll-row {
            display: flex;
            overflow-x: auto;
            gap: 20px;
            padding: 10px 0;
        }
        .scene-card {
            min-width: 260px;
            max-width: 260px;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            background: #fafafa;
        }
        .scene-img {
            width: 100%;
            border-radius: 6px;
        }
        </style>
        """, unsafe_allow_html=True)

        # ============================
        # 横スクロールの HTML を生成
        # ============================
        html = '<div class="scroll-row">'

        for i, s in enumerate(aligned):
            img_html = f'<img src="file://{os.path.abspath(s["img_path"])}" class="scene-img">' if s["img_path"] else ""
            text_html = s["final_text"].replace("\n", "<br>")

            html += f"""
                <div class="scene-card">
                    <div><b>Scene {i+1}</b><br>{s['time_str']}〜</div>
                    {img_html}
                    <div style="margin-top:6px; font-size:13px; white-space:pre-wrap;">{text_html}</div>
                </div>
            """

        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

        # ============================
        # TSV 出力
        # ============================
        st.divider()
        st.subheader("2. スプレッドシート貼り付け用（横一列）")

        tsv = "\t".join([s["final_text"].replace("\n", " ") for s in aligned])
        st.code(tsv, language="text")
