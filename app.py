import streamlit as st
import os
import cv2
import whisper
import shutil
import datetime
# pandasはデータ整形用
import pandas as pd
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# --- 設定 ---
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "temp_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 関数: 時間表示 ---
def format_time(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    rem_seconds = seconds % 60
    return f"{minutes:02}:{rem_seconds:02}"

# --- 関数: フォルダリセット ---
def clear_output_folder():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 関数: シーン抽出 ---
def extract_scenes(video_path):
    # シーン検出器のセットアップ
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    # 動きの感度設定（数字が大きいほど敏感）
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    
    # 画像保存の準備
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    
    scenes_data = []
    
    # シーンリストが空の場合の保険（動画全体を1シーンとする）
    if not scene_list:
        scenes_data.append({
            "start": 0.0,
            "end": duration,
            "time_str": format_time(0),
            "img_path": None
        })
    else:
        # 最初のシーンが0秒から始まっていない場合の補正
        if scene_list[0][0].get_seconds() > 1.0:
            scenes_data.append({
                "start": 0.0,
                "end": scene_list[0][0].get_seconds(),
                "time_str": format_time(0),
                "img_path": None
            })
        
        for scene in scene_list:
            start = scene[0].get_seconds()
            end = scene[1].get_seconds()
            scenes_data.append({
                "start": start,
                "end": end,
                "time_str": format_time(start),
                "img_path": None
            })
    
    # 画像キャプチャ処理
    progress_bar = st.progress(0, text="シーン画像を抽出中...")
    total_scenes = len(scenes_data)
    
    for i, data in enumerate(scenes_data):
        # シーン開始直後より少し後（0.5秒後）を撮ることでブレを防ぐ
        capture_point = data["start"] + 0.5
        if capture_point >= data["end"]:
            capture_point = data["start"] # シーンが短すぎる場合は開始点
            
        cap.set(cv2.CAP_PROP_POS_MSEC, capture_point * 1000)
        ret, frame = cap.read()
        
        if ret:
            img_filename =
