import streamlit as st
import os
import cv2
import whisper
import shutil
import datetime
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
# pandasはテーブル表示のために使用します
import pandas as pd

# --- 設定 ---
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "temp_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 関数: 時間表示 ---
def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

# --- 関数: フォルダリセット ---
def clear_output_folder():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 関数: シーン抽出 ---
def extract_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    # threshold=27.0 は感度の標準値。
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    
    cap = cv2.VideoCapture(video_path)
    scenes_data = []

    progress_bar = st.progress(0, text="シーン検出中...")
    total_scenes = len(scene_list)
    
    # 最初のシーンの開始時間は必ず0秒とする
    if total_scenes > 0 and scene_list[0][0].get_seconds() > 0:
         start_time = 0.0
         cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
         ret, frame = cap.read()
         if ret:
             img_filename = f"scene_start_0s.jpg"
             img_path = os.path.join(OUTPUT_DIR, img_filename)
             cv2.imwrite(img_path, frame)
             scenes_data.append({
                 "time_str": format_time(start_time),
                 "seconds": start_time,
                 "img_path": img_path
             })

    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        ret, frame = cap.read()
        
        if ret:
            img_filename = f"scene_{i:03d}_{int(start_time)}s.jpg"
            img_path = os.path.join(OUTPUT_DIR, img_filename)
            cv2.imwrite(img_path, frame)
            
            scenes_data.append({
                "time_str": format_time(start_time),
                "seconds": start_time,
                "img_path": img_path
            })
        
        if total_scenes > 0:
            progress_bar.progress(min((i + 1) / total_scenes, 1.0))

    cap.release()
    progress_bar.empty()
    return scenes_data

# --- 関数: 音声書き起こし（精度向上版） ---
@st.cache_resource
def load_whisper_model():
    # ★変更点：精度を上げるため "base" から "small" に変更
    # クラウドで落ちる場合は "base" に戻してください。
    # ローカルで余裕があれば "medium" も可。
    return whisper.load_model("small") 

def transcribe_audio(video_path):
    model = load_whisper_model()
    with st.spinner("AIが音声を解析しています... (モデルを大きくしたため時間がかかります)"):
        # language="ja" を指定すると認識率が上がることがあります
        result = model.transcribe(video_path, language="ja")
    return result["segments"]

# --- 関数: データ結合（今回の肝） ---
def align_scenes_and_text(scenes, segments):
    aligned_data = []
    
    for i, scene in enumerate(scenes):
        scene_start = scene["seconds"]
        # 次のシーンの開始時間を取得（最後のシーンの場合は無限大を設定）
        next_scene_start = scenes[i+1]["seconds"] if i+1 < len(scenes) else float('inf')
        
        # このシーンの区間内に開始時間があるテキストセグメントを探す
        matched_texts = []
        for segment in segments:
            if scene_start <= segment["start"] < next_scene_start:
                matched_texts.append(segment["text"])
        
        # 複数行のテキストを結合（スプシで見やすくするため改行を入れる）
        combined_text = "\n".join(matched_texts)
        
        aligned_data.append({
            "time": scene["time_str"],
            "image": scene["img_path"],
            "text": combined_text
        })
    return aligned_data

# ==========================================
# メインUI
# ==========================================
st.set_page_config(page_title="動画解析アプリPro", layout="wide")

st.title("🎥 動画解析アプリ Pro (スプシ対応版)")
st.markdown("""
- **精度向上:** 音声認識モデルを高性能なものに変更しました。
- **スプシ対応:** シーン画像の下に対応するテキストを配置します。そのままコピペできます。
""")

uploaded_file = st.file_uploader("動画ファイルをアップロード", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"読み込み完了: {uploaded_file.name}")

    if st.button("🚀 解析スタート (少し時間がかかります)", type="primary"):
        clear_output_folder()
        
        # 1. 処理実行
        scenes = extract_scenes(video_path)
        segments = transcribe_audio(video_path)
        
        # 2. 画像とテキストの突き合わせ
        aligned_data = align_scenes_and_text(scenes, segments)
        
        st.divider()
        st.subheader("📊 解析結果 (スプレッドシート用レイアウト)")
        st.info("💡 ヒント: 画像の行からテキストの行までドラッグして選択し、ExcelやGoogleスプレッドシートに貼り付けてください。")

        if not aligned_data:
            st.warning("データが抽出できませんでした。")
        else:
            # --- スプシ用レイアウト表示 ---
            # Streamlitで横並びを綺麗にコピペさせるため、少し特殊な表示をします。
            
            num_scenes = len(aligned_data)
            
            # 1行目：時間表示
            cols_time = st.columns(num_scenes)
            for i, col in enumerate(cols_time):
                col.write(f"**{aligned_data[i]['time']}**")
            
            # 2行目：画像表示
            cols_img = st.columns(num_scenes)
            for i, col in enumerate(cols_img):
                col.image(aligned_data[i]["image"], use_column_width=True)
                
            # 3行目：テキスト表示 (テキストエリアを使うとコピペしやすい)
            cols_text = st.columns(num_scenes)
            for i, col in enumerate(cols_text):
                # height調整で見た目を揃える
                col.text_area("テキスト", aligned_data[i]["text"], height=150, label_visibility="hidden", key=f"text_{i}")

            st.divider()
            
            # データフレームでもダウンロードできるようにする
            df = pd.DataFrame(aligned_data)
            csv = df.to_csv(index=False).encode('utf-8_sig')
            st.download_button(
                "📥 CSVでダウンロード",
                csv,
                "video_analysis.csv",
                "text/csv",
                 key='download-csv'
            )
