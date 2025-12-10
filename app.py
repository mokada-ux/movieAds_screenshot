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
    # threshold=27.0 は標準。動きが激しい動画で細切れになる場合は35.0くらいに上げる
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    
    cap = cv2.VideoCapture(video_path)
    # 動画の総再生時間を取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    
    scenes_data = []
    
    # 最初のシーン(0秒地点)を強制追加するか判定
    start_time_offset = 0.0
    if not scene_list or scene_list[0][0].get_seconds() > 1.0:
        scenes_data.append({
            "start": 0.0,
            "end": scene_list[0][0].get_seconds() if scene_list else duration,
            "time_str": format_time(0),
            "img_path": None # 後で撮影
        })

    # シーンリストを整形
    for i, scene in enumerate(scene_list):
        start = scene[0].get_seconds()
        end = scene[1].get_seconds()
        scenes_data.append({
            "start": start,
            "end": end,
            "time_str": format_time(start),
            "img_path": None
        })
    
    # 画像保存処理
    progress_bar = st.progress(0, text="シーン画像を抽出中...")
    total_scenes = len(scenes_data)
    
    for i, data in enumerate(scenes_data):
        # シーン開始直後だとブレていることがあるので、0.5秒後などを取得してみる
        # ただしシーンが短すぎる場合は開始時点を使う
        scene_len = data["end"] - data["start"]
        capture_point = data["start"] + (0.5 if scene_len > 1.0 else 0.0)
        
        cap.set(cv2.CAP_PROP_POS_MSEC, capture_point * 1000)
        ret, frame = cap.read()
        
        if ret:
            img_filename = f"scene_{i:03d}.jpg"
            img_path = os.path.join(OUTPUT_DIR, img_filename)
            cv2.imwrite(img_path, frame)
            scenes_data[i]["img_path"] = img_path
        
        progress_bar.progress(min((i + 1) / total_scenes, 1.0))

    cap.release()
    progress_bar.empty()
    return scenes_data

# --- 関数: 音声書き起こし ---
@st.cache_resource
def load_whisper_model():
    # 精度重視なら small, 更に上げるなら medium
    return whisper.load_model("small")

def transcribe_audio(video_path):
    model = load_whisper_model()
    with st.spinner("AIが音声を解析しています..."):
        result = model.transcribe(video_path, language="ja")
    return result["segments"]

# --- 関数: 精度向上版アライメント（中点ロジック） ---
def align_scenes_and_text(scenes, segments):
    # シーンごとに空のテキストリストを用意
    for scene in scenes:
        scene["text_list"] = []

    for segment in segments:
        # セリフの開始・終了・中間点
        seg_start = segment["start"]
        seg_end = segment["end"]
        seg_mid = (seg_start + seg_end) / 2 # ★ここがポイント

        # 「セリフの中間点」が含まれているシーンを探す
        matched = False
        for scene in scenes:
            # 最後のシーンのend時間が曖昧な場合のガードなどを考慮しつつ判定
            if scene["start"] <= seg_mid < scene["end"]:
                scene["text_list"].append(segment["text"])
                matched = True
                break
        
        # どのシーンにも入らなかった場合（動画最後の余韻など）、最後のシーンに入れる
        if not matched and scenes:
             scenes[-1]["text_list"].append(segment["text"])

    # リストを結合して文字列にする
    for scene in scenes:
        scene["final_text"] = "\n".join(scene["text_list"])
    
    return scenes

# ==========================================
# メインUI
# ==========================================
st.set_page_config(page_title="動画解析アプリ Pro v2", layout="wide")

st.title("🎥 動画解析 & スプシ貼り付けツール")
st.markdown("シーン検出の精度向上と、スプレッドシートへの横並び貼り付けに対応しました。")

uploaded_file = st.file_uploader("動画ファイルをアップロード", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"ファイル準備完了: {uploaded_file.name}")

    if st.button("🚀 解析スタート", type="primary"):
        clear_output_folder()
        
        # 1. 実行
        scenes = extract_scenes(video_path)
        segments = transcribe_audio(video_path)
        
        # 2. 結合（精度向上ロジック適用）
        aligned_data = align_scenes_and_text(scenes, segments)
        
        st.divider()

        # --- 表示エリア ---
        st.subheader("1. 解析結果プレビュー")
        
        # 3列ごとに折り返して表示
        cols = st.columns(3)
        for i, item in enumerate(aligned_data):
            with cols[i % 3]:
                if item["img_path"]:
                    st.image(item["img_path"], use_column_width=True)
                st.caption(f"シーン {i+1} ({item['time_str']}~)")
                st.text_area("内容", item["final_text"], height=100, key=f"t_{i}")

        st.divider()

        # --- スプシ用コピーエリア ---
        st.subheader("2. スプレッドシート貼り付け用データ")
        st.markdown("""
        以下のボックスの右上にある **コピーボタン** を押してください。  
        その後、スプレッドシートのセルを選んで貼り付けると、**横一列にシーンごとのテキストが入ります。**
        """)

        # タブ区切りテキスト(TSV)を作成
        # joinするときにタブ(\t)を使うことで、エクセル等は「隣のセル」と認識します
        tsv_text = "\t".join([item["final_text"].replace("\n", " ") for item in aligned_data])
        
        # st.codeを使ってコピーボタン付きのボックスを表示
        st.code(tsv_text, language="text")
