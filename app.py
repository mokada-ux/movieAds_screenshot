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
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    
    scenes_data = []
    
    if not scene_list:
        scenes_data.append({
            "start": 0.0,
            "end": duration,
            "time_str": format_time(0),
            "img_path": None
        })
    else:
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
    
    progress_bar = st.progress(0, text="シーン画像を抽出中...")
    total_scenes = len(scenes_data)
    
    for i, data in enumerate(scenes_data):
        capture_point = data["start"] + 0.5
        if capture_point >= data["end"]:
            capture_point = data["start"]
            
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
    return whisper.load_model("base")

def transcribe_audio(video_path):
    model = load_whisper_model()
    with st.spinner("AIが音声を解析しています..."):
        result = model.transcribe(video_path, language="ja")
    return result["segments"]

# --- 関数: 結合ロジック ---
def align_scenes_and_text(scenes, segments):
    for scene in scenes:
        scene["text_list"] = []

    for segment in segments:
        mid_point = (segment["start"] + segment["end"]) / 2
        matched = False
        for scene in scenes:
            if scene["start"] <= mid_point < scene["end"]:
                scene["text_list"].append(segment["text"])
                matched = True
                break
        if not matched and scenes:
            scenes[-1]["text_list"].append(segment["text"])

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

            # --- A. プレビュー表示（レイアウト変更版） ---
            st.subheader("1. 解析結果プレビュー")
            
            # 1行に表示するシーン数（ここを変えると画像の大きさが変わります）
            ITEMS_PER_ROW = 15 
            
            # データを分割して表示ループ
            for i in range(0, len(aligned_data), ITEMS_PER_ROW):
                # 今回表示するバッチ（最大8個）
                batch = aligned_data[i : i + ITEMS_PER_ROW]
                cols_count = len(batch)
                
                # 1段目：画像 (スクショ)
                cols_img = st.columns(cols_count)
                for j, col in enumerate(cols_img):
                    if batch[j]["img_path"]:
                        col.image(batch[j]["img_path"], use_column_width=True)
                
                # 2段目：時間 (秒数)
                cols_time = st.columns(cols_count)
                for j, col in enumerate(cols_time):
                    # 中央揃えっぽく見せるためにmarkdownを使用
                    col.markdown(f"**{batch[j]['time_str']}**")
                
                # 3段目：テキスト
                cols_text = st.columns(cols_count)
                for j, col in enumerate(cols_text):
                    # テキストエリアの高さを小さくして一覧性を高める
                    col.text_area("text", batch[j]["final_text"], height=100, label_visibility="collapsed", key=f"txt_{i}_{j}")
                
                # 区切り線
                st.divider()

            # --- B. スプシ貼り付け用データ ---
            st.subheader("2. スプレッドシート貼り付け用データ")
            st.info("👇 下のボックスの右上にあるコピーボタンを押し、スプレッドシートのA1セルを選択して貼り付けてください。横一列に展開されます。")

            tsv_list = []
            for item in aligned_data:
                clean_text = item["final_text"].replace("\n", " ").replace("\t", " ")
                tsv_list.append(clean_text)
            
            tsv_string = "\t".join(tsv_list)
            
            st.code(tsv_string, language="text")
            
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
