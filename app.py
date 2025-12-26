import streamlit as st
import os
import cv2
import whisper
import shutil
import datetime
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
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    
    # 0.5秒(15フレーム)程度あればシーンとみなす
    scene_manager.add_detector(ContentDetector(threshold=27.0, min_scene_len=15))
    
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    
    scenes_data = []
    
    # シーンがない場合
    if not scene_list:
        scenes_data.append({"start": 0.0, "end": duration})
    else:
        # 開始地点の補正
        if scene_list[0][0].get_seconds() > 0.5:
             scenes_data.append({"start": 0.0, "end": scene_list[0][0].get_seconds()})
        
        for scene in scene_list:
            scenes_data.append({
                "start": scene[0].get_seconds(),
                "end": scene[1].get_seconds()
            })
            
    final_scenes = []
    progress_bar = st.progress(0, text="シーン抽出中...")
    total_scenes = len(scenes_data)

    for i, scene in enumerate(scenes_data):
        start = scene["start"]
        end = scene["end"]
        
        scene_item = {
            "start": start,
            "end": end,
            "time_str": format_time(start),
            "img_path": None,
            "text_list": [] 
        }

        # サムネイル位置調整
        capture_point = start + min((end - start) / 2, 1.0)
        
        cap.set(cv2.CAP_PROP_POS_MSEC, capture_point * 1000)
        ret, frame = cap.read()
        
        if ret:
            img_filename = f"scene_{i:03d}.jpg"
            img_path = os.path.join(OUTPUT_DIR, img_filename)
            cv2.imwrite(img_path, frame)
            scene_item["img_path"] = img_path
            final_scenes.append(scene_item)
        
        if total_scenes > 0:
            progress_bar.progress(min((i + 1) / total_scenes, 1.0))

    cap.release()
    progress_bar.empty()
    return final_scenes

# --- 関数: 音声書き起こし (日本語固定) ---
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def transcribe_audio(video_path):
    model = load_whisper_model()
    
    initial_prompt = "ここには動画の音声が含まれています。フィラーを除去して、正確な日本語の文章に書き起こしてください。"
    
    with st.spinner("AIが音声を解析中..."):
        # task="transcribe" (日本語固定)
        result = model.transcribe(
            video_path, 
            language="ja",
            initial_prompt=initial_prompt,
            condition_on_previous_text=False
        )
    return result["segments"]

# --- 関数: 結合ロジック（修正版：一点集中型） ---
def align_scenes_and_text(scenes, segments):
    # 初期化
    for scene in scenes:
        scene["text_list"] = []

    for segment in segments:
        seg_start = segment["start"]
        seg_end = segment["end"]
        
        # セリフの「真ん中の時間」を計算
        mid_point = (seg_start + seg_end) / 2
        
        # 真ん中の時間が含まれるシーン「1つだけ」に割り当てる
        matched = False
        for scene in scenes:
            if scene["start"] <= mid_point < scene["end"]:
                scene["text_list"].append(segment["text"])
                matched = True
                break # 1つ見つかったら他には入れない（重複防止）
        
        # もしどのシーンにも当てはまらなかったら（動画末尾など）、最後のシーンへ
        if not matched and scenes:
            scenes[-1]["text_list"].append(segment["text"])

    # リスト結合
    for scene in scenes:
        scene["final_text"] = "\n".join(scene["text_list"])
    
    return scenes

# ==========================================
# メインUI
# ==========================================
st.set_page_config(page_title="動画解析アプリ Simple", layout="wide")

st.title("🎥 動画解析 & スプシ一括貼り付け")

# サイドバー削除完了。シンプルUIへ。

uploaded_file = st.file_uploader("動画ファイルをアップロード (MP4/MOV)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"準備完了: {uploaded_file.name}")

    if st.button("🚀 解析スタート", type="primary"):
        clear_output_folder()
        
        try:
            # 1. 解析
            scenes = extract_scenes(video_path)
            segments = transcribe_audio(video_path) # 引数なし(日本語固定)
            
            # 2. 結合
            aligned_data = align_scenes_and_text(scenes, segments)
            
            st.divider()

            # --- A. プレビュー表示 (8列) ---
            st.subheader("1. 解析結果プレビュー")
            
            ITEMS_PER_ROW = 8
            
            for i in range(0, len(aligned_data), ITEMS_PER_ROW):
                batch = aligned_data[i : i + ITEMS_PER_ROW]
                cols_count = len(batch)
                
                # 画像
                cols_img = st.columns(cols_count)
                for j, col in enumerate(cols_img):
                    if batch[j]["img_path"]:
                        col.image(batch[j]["img_path"], use_column_width=True)
                
                # 時間
                cols_time = st.columns(cols_count)
                for j, col in enumerate(cols_time):
                    col.markdown(f"**{batch[j]['time_str']}**")
                
                # テキスト
                cols_text = st.columns(cols_count)
                for j, col in enumerate(cols_text):
                    val = batch[j]["final_text"]
                    # 何もないときは空欄
                    col.text_area("text", val, height=100, label_visibility="collapsed", key=f"txt_{i}_{j}")
                
                st.divider()

            # --- B. スプシ貼り付け用データ ---
            st.subheader("2. スプレッドシート貼り付け用データ")
            
            tsv_list = []
            for item in aligned_data:
                clean_text = item["final_text"].replace("\n", " ").replace("\t", " ")
                tsv_list.append(clean_text)
            
            tsv_string = "\t".join(tsv_list)
            
            st.code(tsv_string, language="text")
            
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
