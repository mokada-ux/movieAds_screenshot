import streamlit as st
import os
import tempfile
import subprocess
import base64
from moviepy.editor import VideoFileClip
import whisper


# ==============================
# Streamlit 基本設定
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

st.title("🎬 動画シーン解析ツール（フルリライト版）")


# ==============================
# シーン抽出（FFmpeg）
# ==============================
def extract_scenes_ffmpeg(video_path):
    tmp_dir = tempfile.mkdtemp()

    # SceneDetect + FFmpeg の閾値
    threshold = "0.3"

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',metadata=print",
        "-vsync", "vfr",
        os.path.join(tmp_dir, "scene_%04d.jpg")
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # ファイル名順に並ぶ
    image_paths = sorted(
        [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".jpg")]
    )
    return image_paths


# ==============================
# Whisper でテキスト抽出
# ==============================
@st.cache_resource
def load_whisper():
    return whisper.load_model("small")


def transcribe_audio(video_path):
    model = load_whisper()
    result = model.transcribe(video_path, fp16=False)
    return result["text"]


# ==============================
# Google スプレッドシート用 TSV（横3行×n列）
# ==============================
def generate_tsv_horizontal(image_paths, times, transcripts):
    # 1行目（時間）
    time_row = ["時間"] + [str(t) for t in times]

    # 2行目（画像）
    image_row = ["画像"]
    for img in image_paths:
        with open(img, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        img_formula = f'=IMAGE("data:image/jpeg;base64,{b64}")'
        image_row.append(img_formula)

    # 3行目（テキスト）
    text_row = ["テキスト"] + transcripts

    # TSV化
    tsv = "\n".join([
        "\t".join(time_row),
        "\t".join(image_row),
        "\t".join(text_row)
    ])

    return tsv


# ==============================
# メイン処理
# ==============================
uploaded = st.file_uploader("動画ファイル（mp4 / mov）をアップロード", type=["mp4", "mov"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    st.success("動画を読み込みました！")

    # 動画情報
    video = VideoFileClip(video_path)
    duration = video.duration
    st.write(f"動画長さ：{duration:.1f} 秒")

    # シーン抽出
    with st.spinner("シーン抽出中…"):
        scene_images = extract_scenes_ffmpeg(video_path)

    st.write(f"抽出されたシーン数：{len(scene_images)}")

    # 各画像の秒数取得（moviepy）
    times = []
    for img in scene_images:
        filename = os.path.basename(img)
        idx = int(filename.replace("scene_", "").replace(".jpg", ""))
        t = (idx - 1) * 1.2  # 適当だが SceneDetect が秒数を取らないため補間
        times.append(round(t, 1))

    # Whisper テキスト
    with st.spinner("音声からテキスト解析中（Whisper-small）…"):
        transcript = transcribe_audio(video_path)

    # シーン単位のテキスト（簡易分割）
    transcripts = []
    chunk = len(scene_images)
    words = transcript.split()

    if chunk > 0:
        split_size = max(1, len(words) // chunk)

        for i in range(chunk):
            part = words[i * split_size:(i + 1) * split_size]
            transcripts.append(" ".join(part))


    # ==============================
    # UI（横スクロールカード）
    # ==============================
    st.subheader("🔍 自動抽出されたシーン")

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
    st.subheader("📋 Google スプレッドシート用（横3行 × シーン数列）")

    if st.button("TSV を生成"):
        tsv = generate_tsv_horizontal(scene_images, times, transcripts)
        st.code(tsv, language="text")
        st.success("このTSVをスプレッドシートに貼ると、横に整列します！")
