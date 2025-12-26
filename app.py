import streamlit as st
import os
import cv2
import whisper
import shutil
import datetime
import numpy as np
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

# --- 関数: シーン抽出（改良版） ---
def extract_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    
    # threshold=30.0 に上げて、細かい光の変化での誤検知を減らす
    # min_scene_len=30 (約1秒) 以下の細かいカットを無視する
    scene_manager.add_detector(ContentDetector(threshold=30.0, min_scene_len=30))
    
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    
    raw_scenes = []
    
    # シーンがない場合は全体を1シーンとする
    if not scene_list:
        raw_scenes.append({"start": 0.0, "end": duration})
    else:
        # 最初のシーンの補正
        if scene_list[0][0].get_seconds() > 1.0:
            raw_scenes.append({"start": 0.0, "end": scene_list[0][0].get_seconds()})
        
        for scene in scene_list:
            raw_scenes.append({
                "start": scene[0].get_seconds(),
                "end": scene[1].get_seconds()
            })

    # 【重要】短すぎるシーン（1.5秒未満）を結合してノイズを減らす処理
    merged_scenes = []
    if raw_scenes:
        current_scene = raw_scenes[0]
        for next_scene in raw_scenes[1:]:
            # シーンが1.5秒より短い場合、強制的に前のシーンと繋げる
            if (current_scene["end"] - current_scene["start"]) < 1.5:
                current_scene["end"] = next_scene["end"]
            else:
                merged_scenes.append(current_scene)
                current_scene = next_scene
        merged_scenes.append(current_scene)

    # データの整形と画像保存
    scenes_data = []
    progress_bar = st.progress(0, text="シーン画像を抽出中...")
    total_scenes = len(merged_scenes)

    for i, scene in enumerate(merged_scenes):
        start = scene["start"]
        end = scene["end"]
        
        scenes_data.append({
            "start": start,
            "end": end,
            "time_str": format_time(start),
            "img_path": None,
            "text_list": [] # テキスト格納用
        })

        # サムネイル取得（シーンの開始地点だとブレるので、少し進める）
        # ただしシーンの長さの範囲内に収める
        mid_point = start + min((end - start) / 2, 2.0) # 最大でも開始から2秒地点
        
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_point * 1000)
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

# --- 関数: 音声書き起こし（チューニング版） ---
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def transcribe_audio(video_path):
    model = load_whisper_model()
    with st.spinner("AIが音声を解析しています..."):
        # condition_on_previous_text=False: 前の文脈による幻覚（ループ）を防ぐ
        # temperature=0.0: 毎回同じ結果が出るように固定（ランダム性を排除）
        result = model.transcribe(
            video_path, 
            language="ja", 
            condition_on_previous_text=False,
            temperature=0.0
        )
    return result["segments"]

# --- 関数: 結合ロジック（最大重複判定） ---
def align_scenes_and_text(scenes, segments):
    # すべてのテキストセグメントに対して
    for segment in segments:
        seg_start = segment["start"]
        seg_end = segment["end"]
        seg_duration = seg_end - seg_start
        
        if seg_duration <= 0:
            continue

        best_scene_index = -1
        max_overlap = 0.0

        # すべてのシーンと突き合わせる
        for i, scene in enumerate(scenes):
            scene_start = scene["start"]
            scene_end = scene["end"]

            # 重なっている期間(秒)を計算
            overlap_start = max(seg_start, scene_start)
            overlap_end = min(seg_end, scene_end)
            overlap = max(0, overlap_end - overlap_start)

            # 最も長く重なっているシーンを探す
            if overlap > max_overlap:
                max_overlap = overlap
                best_scene_index = i
        
        # 重なりが見つかった場合、そのシーンにテキストを追加
        # もし重なりがゼロなら（シーンの切れ目など）、開始時間が含まれるシーンに入れる
        if best_scene_index != -1:
             scenes[best_scene_index]["text_list"].append(segment["text"])
        else:
            # フォールバック：開始時間で判定
            for i, scene in enumerate(scenes):
                if scene["start"] <= seg_start < scene["end"]:
                    scenes[i]["text_list"].append(segment["text"])
                    break

    # リストを結合
    for scene in scenes:
        scene["final_text"] = "\n".join(scene["text_list"])
    
    return scenes

# ==========================================
# メインUI
# ==========================================
st.set_page_config(page_title="動画解析アプリ Pro Cloud", layout="wide")

st.title("🎥 動画解析 & スプシ一括貼り付け")
st.markdown("精度改善版：短いシーンを結合し、音声の重複判定を強化しました。")

uploaded_file = st.file_uploader("動画ファイルをアップロード (MP4推奨)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
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

            # --- A. プレビュー表示（サムネイル縮小・多列表示） ---
            st.subheader("1. 解析結果プレビュー")
            
            # 横に並べる数（数を増やすと画像が小さくなります）
            ITEMS_PER_ROW = 15
            
            for i in range(0, len(aligned_data), ITEMS_PER_ROW):
                batch = aligned_data[i : i + ITEMS_PER_ROW]
                cols_count = len(batch)
                
                # 1段目：画像
                cols_img = st.columns(cols_count)
                for j, col in enumerate(cols_img):
                    if batch[j]["img_path"]:
                        col.image(batch[j]["img_path"], use_column_width=True)
                
                # 2段目：時間
                cols_time = st.columns(cols_count)
                for j, col in enumerate(cols_time):
                    col.markdown(f"**{batch[j]['time_str']}**")
                
                # 3段目：テキスト
                cols_text = st.columns(cols_count)
                for j, col in enumerate(cols_text):
                    col.text_area("text", batch[j]["final_text"], height=100, label_visibility="collapsed", key=f"txt_{i}_{j}")
                
                st.divider()

            # --- B. スプシ貼り付け用データ ---
            st.subheader("2. スプレッドシート貼り付け用データ")
            st.info("👇 下のボックスの右上にあるコピーボタンを押し、スプレッドシートのA1セルを選択して貼り付けてください。")

            tsv_list = []
            for item in aligned_data:
                clean_text = item["final_text"].replace("\n", " ").replace("\t", " ")
                tsv_list.append(clean_text)
            
            tsv_string = "\t".join(tsv_list)
            
            st.code(tsv_string, language="text")
            
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
