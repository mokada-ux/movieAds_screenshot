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
                end_time = frame_idx
