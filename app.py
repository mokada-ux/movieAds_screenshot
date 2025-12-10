# app.py (フルリライト)
import os
import io
import math
import shutil
import zipfile
import base64
import tempfile
from typing import List, Dict, Tuple

import streamlit as st
from PIL import Image
import cv2
import whisper
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# --------------------
# 設定
# --------------------
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "temp_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="動画解析 Pro — Stable UI", layout="wide")

# --------------------
# ユーティリティ
# --------------------
def format_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s//60:02}:{s%60:02}"

def clear_output_folder():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# キャッシュ: base64 変換（ファイルパス単位でキャッシュ）
# --------------------
@st.cache_data(show_spinner=False)
def load_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# --------------------
# Whisper モデルロード（キャッシュ）
# --------------------
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str = "small"):
    # model_size: "tiny", "small", "medium", "large" など
    return whisper.load_model(model_size)

# --------------------
# シーン抽出（SceneDetect -> fallback frame-diff）
# --------------------
def extract_scenes_with_scenedetect(video_path: str, threshold: float = 27.0) -> List[Dict]:
    try:
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        # video_manager.release()  # VideoManager deprec warning; cap.release used below
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / fps if fps else 0
        scenes = []
        # 首シーン補正
        if not scene_list or scene_list[0][0].get_seconds() > 1.0:
            scenes.append({"start": 0.0, "end": scene_list[0][0].get_seconds() if scene_list else duration})
        for s in scene_list:
            scenes.append({"start": s[0].get_seconds(), "end": s[1].get_seconds()})
        cap.release()
        return scenes
    except Exception as e:
        st.warning(f"SceneDetect に失敗しました（fallback を実行します）: {e}")
        return []

def fallback_extract_scenes_by_diff(video_path: str, threshold: float = 30.0, min_scene_len: float = 0.8) -> List[Dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    last_gray = None
    frame_idx = 0
    start_time = 0.0
    scenes = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if last_gray is not None:
            diff = cv2.absdiff(gray, last_gray)
            score = float(diff.mean())
            if score > threshold:
                end_time = frame_idx / fps
                if end_time - start_time >= min_scene_len:
                    scenes.append({"start": start_time, "end": end_time})
                start_time = end_time
        last_gray = gray
        frame_idx += 1
    # 最後のシーンを加える
    total_duration = (frame_idx / fps) if fps else 0
    if total_duration - start_time >= 0.1:
        scenes.append({"start": start_time, "end": total_duration})
    cap.release()
    return scenes

def extract_scenes(video_path: str, threshold: float = 27.0) -> List[Dict]:
    scenes = extract_scenes_with_scenedetect(video_path, threshold=threshold)
    if not scenes:
        scenes = fallback_extract_scenes_by_diff(video_path, threshold=threshold)
    # ensure at least one scene
    if not scenes:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / fps if fps else 0
        scenes = [{"start": 0.0, "end": duration}]
        cap.release()
    # attach time_str placeholders
    for sc in scenes:
        sc["time_str"] = format_time(sc["start"])
        sc["img"] = None
        sc["text"] = ""
    return scenes

# --------------------
# スクショ取得（シーン中点）
# --------------------
def capture_frame(video_path: str, time_sec: float) -> Image.Image | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, int(time_sec * 1000))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

# --------------------
# Whisper 書き起こし
# --------------------
def transcribe_video(video_path: str, model_size: str = "small", language: str = "ja"):
    model = load_whisper_model(model_size)
    # Whisper の transcribe はファイルパスを渡せる
    result = model.transcribe(video_path, language=language)
    return result.get("segments", [])

# --------------------
# セグメント -> シーン アライメント（中点）
# --------------------
def align_scenes_and_segments(scenes: List[Dict], segments: List[Dict]) -> List[Dict]:
    for sc in scenes:
        sc["text_list"] = []
    for seg in segments:
        seg_mid = (seg["start"] + seg["end"]) / 2
        matched = False
        for sc in scenes:
            if sc["start"] <= seg_mid < sc["end"]:
                sc["text_list"].append(seg.get("text", "").strip())
                matched = True
                break
        if not matched and scenes:
            scenes[-1]["text_list"].append(seg.get("text", "").strip())
    for sc in scenes:
        sc["text"] = "\n".join([t for t in sc.get("text_list", []) if t])
    return scenes

# --------------------
# UI: CSS
# --------------------
st.markdown(
    """
    <style>
    .scene-container { display:flex; gap:18px; overflow-x:auto; padding:12px 8px 20px 8px; }
    .scene-card { min-width:300px; max-width:320px; background: #fff; border-radius:12px; padding:12px; box-shadow:0 6px 18px rgba(0,0,0,0.08); }
    .scene-img { width:100%; height:auto; border-radius:8px; display:block; }
    .scene-meta { font-size:13px; color:#555; margin-top:8px; }
    .scene-text { margin-top:8px; white-space:pre-wrap; font-size:14px; line-height:1.5; color:#222; max-height:220px; overflow:auto; }
    .controls { display:flex; gap:8px; align-items:center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------
# Sidebar: 設定
# --------------------
st.sidebar.header("解析設定")
model_choice = st.sidebar.selectbox("Whisper model", options=["small", "medium"], index=0,
                                    help="small:安定 / medium:より高精度（重い）")
threshold = st.sidebar.slider("シーン検出しきい値 (diff/ContentDetector)", 15, 60, 27)
max_minutes = st.sidebar.number_input("最大許可動画時間（分）", min_value=1, max_value=60, value=20)
allow_zip = st.sidebar.checkbox("画像をZIPでダウンロード可能にする", value=True)

st.title("🎞️ 動画シーン抽出 + 高精度文字起こし (ローカルWhisper)")
st.caption("画像は横スクロール表示。画像の下にシーンごとの書き起こしを表示します。")

# --------------------
# ファイルアップロード
# --------------------
uploaded = st.file_uploader("動画ファイルをアップロード（.mp4/.mov 等）", type=["mp4", "mov", "avi", "mkv"])
if not uploaded:
    st.info("まずは動画をアップロードしてください。")
    st.stop()

# 一時保存（確実にファイルパスが必要）
with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmpf:
    tmpf.write(uploaded.getbuffer())
    video_path = tmpf.name

# ファイルサイズ / 長さチェック（長すぎると処理が重い）
try:
    cap_check = cv2.VideoCapture(video_path)
    fps_check = cap_check.get(cv2.CAP_PROP_FPS) or 30.0
    frames_check = cap_check.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_sec = frames_check / fps_check if fps_check else 0
    cap_check.release()
except Exception:
    duration_sec = 0

if duration_sec and duration_sec > max_minutes * 60:
    st.warning(f"動画が {max_minutes} 分を超えています（{math.ceil(duration_sec/60)} 分）。解析を続けますか？")
    if not st.button("続行する（自己責任）"):
        st.stop()

# --------------------
# 実行ボタン
# --------------------
if st.button("🚀 解析スタート", type="primary"):
    clear_output_folder()
    st.info("① シーン抽出を行います...")
    with st.spinner("シーン抽出中..."):
        scenes = extract_scenes(video_path, threshold=threshold)
        if not scenes:
            # fallback
            scenes = fallback_extract_scenes_by_diff(video_path, threshold=threshold)
        # ensure at least one
        if not scenes:
            st.error("シーン抽出に失敗しました。")
            st.stop()

    st.success(f"{len(scenes)} シーンを検出しました。")

    # キャプチャ画像を作る
    st.info("② シーン画像を作ります...")
    for i, sc in enumerate(scenes):
        mid = (sc["start"] + sc["end"]) / 2
        img = capture_frame(video_path, mid)
        if img:
            out_path = os.path.join(OUTPUT_DIR, f"scene_{i:03d}.jpg")
            img.save(out_path, format="JPEG", quality=80)
            sc["img"] = out_path
        else:
            sc["img"] = None

    st.success("画像作成完了。")

    # Whisper 書き起こし
    st.info("③ Whisper で音声の文字起こしを行います...")
    try:
        segments = transcribe_video(video_path, model_size=model_choice, language="ja")
    except Exception as e:
        st.error(f"Whisper の実行でエラーが発生しました: {e}")
        st.stop()

    st.success(f"書き起こし完了（セグメント数: {len(segments)}）")

    # アライメント
    scenes = align_scenes_and_segments(scenes, segments)

    # --------------------
    # ギャラリー表示（横スクロール）
    # --------------------
    st.subheader("🔎 シーン一覧（横スクロール）")
    html = '<div class="scene-container">'
    for idx, sc in enumerate(scenes):
        img_b64 = ""
        if sc.get("img") and os.path.exists(sc["img"]):
            img_b64 = load_image_b64(sc["img"])
            text_html = st.markdown  # placeholder to satisfy linter (unused)
            text_escaped = sc.get("text", "").replace("\n", "<br>")
            html += f"""
            <div class="scene-card">
                <img src="data:image/jpeg;base64,{img_b64}" class="scene-img" />
                <div class="scene-meta"><b>Scene {idx+1}</b> &nbsp; {sc['time_str']}〜</div>
                <div class="scene-text">{text_escaped}</div>
            </div>
            """
        else:
            html += f"""
            <div class="scene-card">
                <div style="height:160px; display:flex;align-items:center;justify-content:center;background:#f6f6f6;border-radius:8px;">No Image</div>
                <div class="scene-meta"><b>Scene {idx+1}</b> &nbsp; {sc['time_str']}〜</div>
                <div class="scene-text">{sc.get('text','')}</div>
            </div>
            """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # --------------------
    # TSV と ZIP ダウンロード
    # --------------------
    st.subheader("📥 出力ダウンロード")
    tsv_text = "\t".join([ (s.get("text","").replace("\n", " ") if s.get("text") else "") for s in scenes ])
    st.code(tsv_text, language="text")

    if allow_zip:
        # 画像をZIPにまとめてバイト配列にする
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for s in scenes:
                if s.get("img") and os.path.exists(s["img"]):
                    zf.write(s["img"], arcname=os.path.basename(s["img"]))
        zip_buf.seek(0)
        st.download_button("画像をZIPでダウンロード", zip_buf, file_name="scenes_images.zip", mime="application/zip")

    # TSVダウンロード
    tsv_bytes = tsv_text.encode("utf-8")
    st.download_button("TSVをダウンロード（横並び）", tsv_bytes, file_name="scenes_texts.tsv", mime="text/tab-separated-values")

    st.success("完了！必要なら次に以下をやります:\n・カードクリックで拡大表示\n・画像のトリミング/補正\n・WhisperをAPI化して高速化（有料）")
