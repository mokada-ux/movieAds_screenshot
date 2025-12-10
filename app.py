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
            img_filename = f"scene_{i:03d}.jpg"
            img_path = os.path.join(OUTPUT_DIR, img_filename)
            cv2.imwrite(img_path, frame)
            scenes_data[i]["img_path"] = img_path
        
        if total_scenes > 0:
            progress_bar.progress(min((i + 1) / total_scenes, 1.0))

    cap.release()
    progress_bar.empty()
    return scenes_data

# --- 関数: 音声書き起こし ---
@st.cache_resource
def load_whisper_model():
    # クラウド環境のメモリ制限対策として "base" を使用
    return whisper.load_model("base")

def transcribe_audio(video_path):
    model = load_whisper_model()
    with st.spinner("AIが音声を解析しています..."):
        # 日本語指定で精度アップ
        result = model.transcribe(video_path, language="ja")
    return result["segments"]

# --- 関数: 結合ロジック（中点合わせ） ---
def align_scenes_and_text(scenes, segments):
    # シーンごとにテキストリストを初期化
    for scene in scenes:
        scene["text_list"] = []

    for segment in segments:
        # セリフの中間時間を計算
        mid_point = (segment["start"] + segment["end"]) / 2
        
        # 中間時間がどのシーンに含まれるか判定
        matched = False
        for scene in scenes:
            if scene["start"] <= mid_point < scene["end"]:
                scene["text_list"].append(segment["text"])
                matched = True
                break
        
        # どこにも属さなかった場合（末尾など）、最後のシーンへ
        if not matched and scenes:
            scenes[-1]["text_list"].append(segment["text"])

    # リストを結合
    for scene in scenes:
        scene["final_text"] = "\n".join(scene["text_list"])
    
    return scenes

# ==========================================
# メインUI
# ==========================================
st.set_page_config(page_title="動画解析アプリ Pro Cloud", layout="wide")

st.title("🎥 動画解析 & スプシ一括貼り付け")
st.markdown("Streamlit Cloud対応版：シーン画像抽出と文字起こしを行い、Excel/スプレッドシートへの貼り付け用データを作成します。")

uploaded_file = st.file_uploader("動画ファイルをアップロード (MP4推奨)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # 一時保存
    video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"準備完了: {uploaded_file.name}")

    if st.button("🚀 解析スタート", type="primary"):
        clear_output_folder()
        
        try:
            # 1. 解析実行
            scenes = extract_scenes(video_path)
            segments = transcribe_audio(video_path)
            
            # 2. データ結合
            aligned_data = align_scenes_and_text(scenes, segments)
            
            st.divider()

            # --- A. プレビュー表示 ---
            st.subheader("1. 解析結果プレビュー")
            cols = st.columns(3)
            for i, item in enumerate(aligned_data):
                with cols[i % 3]:
                    if item["img_path"]:
                        st.image(item["img_path"], use_column_width=True)
                    st.caption(f"シーン {i+1} ({item['time_str']}~)")
                    st.text(item["final_text"])

            st.divider()

            # --- B. スプシ貼り付け用データ ---
            st.subheader("2. スプレッドシート貼り付け用データ")
            st.info("👇 下のボックスの右上にあるコピーボタンを押し、スプレッドシートのA1セルを選択して貼り付けてください。横一列に展開されます。")

            # タブ区切りデータを作成 (改行はスペースに置換してセル崩れを防止)
            tsv_list = []
            for item in aligned_data:
                clean_text = item["final_text"].replace("\n", " ").replace("\t", " ")
                tsv_list.append(clean_text)
            
            tsv_string = "\t".join(tsv_list)
            
            st.code(tsv_string, language="text")
            
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
