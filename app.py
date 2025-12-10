import streamlit as st
import os
import tempfile
import subprocess
import base64
import whisper

# ==============================
# Streamlit 設定
# ==============================
st.set_page_config(page_title="動画シーン解析ツール", layout="wide")
st.markdown("""
<style>
.scene-container {
    display: flex;
    flex-wrap: nowrap;
    overflow-x: auto;
    gap: 20px;
    padding: 10px;
}
.scene-card {
    flex: 0 0 auto;
    width: 260px;
    background: #ffffff10;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid #888;
}
.scene-img {
    width: 100%;
    border-radius: 8px;
    border: 1px solid #666;
}
.scene-time {
    font-size: 14px;
    margin-top: 6px;
    color: #ddd;
}
.scene-text {
    font-size: 15px;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 動画シーン解析ツール（MacBook向けリライト版）")

# ==============================
# 動画秒数取得
# ==============================
def get_video_duration(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return float(result.stdout.decode().strip())

# ==============================
# シーン抽出（FFmpeg）
# ==============================
def extract_scenes_ffmpeg(video_path):
    tmp_dir = tempfile.mkdtemp()
    threshold = "0.3"
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf",
        f"select='gt(scene,{threshold})',metadata=print",
        "-vsync", "vfr",
        os.path.join(tmp_dir, "scene_%04d.jpg")
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    image_paths = sorted(
        [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".jpg")]
    )
    return image_paths

# ==============================
# Whisper-small 音声解析
# ==============================
@st.cache_resource
def load_whisper():
    return whisper.load_model("small")

def transcribe_audio(video_path):
    model = load_whisper()
    result = model.transcribe(video_path, fp16=False, language="ja")
    return result["text"]

# ==============================
# TSV（横3行×n列）
# ==============================
def generate_tsv_horizontal(image_paths, times, transcripts):
    time_row = ["時間"] + [str(t) for t in times]
    image_row = ["画像"]
    for img in image_paths:
        with open(img, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        formula = f'=IMAGE("data:image/jpeg;base64,{b64}")'
        image_row.append(formula)
    text_row = ["テキスト"] + transcripts
    return "\n".join([
        "\t".join(time_row),
        "\t".join(image_row),
        "\t".join(text_row)
    ])

# ==============================
# メイン処理
# ==============================
uploaded = st.file_uploader(
    "動画ファイル（mp4 / mov）をアップロード", type=["mp4", "mov"]
)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    st.success("動画を読み込みました！")

    duration = get_video_duration(video_path)
    st.write(f"動画の長さ：{duration:.1f} 秒")

    with st.spinner("シーン抽出中…"):
        scene_images = extract_scenes_ffmpeg(video_path)
    st.write(f"抽出シーン数：{len(scene_images)}")

    # シーン秒数（均等割り）
    times = []
    for i, img in enumerate(scene_images):
        sec = round(i * (duration / max(1, len(scene_images))), 1)
        times.append(sec)

    # Whisper-small
    with st.spinner("音声解析中…"):
        transcript = transcribe_audio(video_path)

    # 均等に分割
    words = transcript.split()
    chunk = len(scene_images)
    transcripts = []
    if chunk > 0:
        split_size = max(1, len(words) // chunk)
        for i in range(chunk):
            part = words[i * split_size:(i + 1) * split_size]
            transcripts.append(" ".join(part))

    # ==============================
    # UI（横スクロールカード）
    # ==============================
    st.subheader("🔍 自動抽出シーン（横スクロール）")
    html = '<div class="scene-container">'
    for img, t, tx in zip(scene_images, times, transcripts):
        with open(img, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        html += f"""
        <div class="scene-card">
            <img class="scene-img" src="data:image/jpeg;base64,{b64}">
            <div class="scene-time">⏱ {t} 秒</div>
            <div class="scene-text">{tx}</div>
        </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # ==============================
    # TSV出力
    # ==============================
    st.subheader("📋 Googleスプレッドシート用 TSV（横3行×n列）")
    if st.button("TSV を生成"):
        tsv = generate_tsv_horizontal(scene_images, times, transcripts)
        st.code(tsv, language="text")
        st.success("そのままスプレッドシートに貼り付け可能！")
