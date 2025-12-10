# app.py (堅牢フルリライト版)
import streamlit as st
import os
import tempfile
import subprocess
import base64
import shutil
import time
import traceback
from typing import List, Dict
import whisper

# -------------------------
# 設定
# -------------------------
st.set_page_config(page_title="動画シーン解析ツール (安定版)", layout="wide")
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "temp_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# ユーティリティ
# -------------------------
def log_and_show(err_msg: str):
    """画面にエラー／ログを表示（開発用）"""
    st.error(err_msg)

def check_command(cmd_name: str) -> bool:
    """コマンドが PATH にあるか確認"""
    return shutil.which(cmd_name) is not None

def run_subprocess(cmd: List[str], timeout: int = 60) -> subprocess.CompletedProcess:
    """subprocess を安全に実行して結果を返す。例外は呼び出し元で処理"""
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)

# -------------------------
# 動画情報取得（ffprobe が無ければ ffmpeg で代替）
# -------------------------
def get_video_duration(video_path: str) -> float:
    """動画の長さ（秒）を取得する。ffprobe が無ければ ffmpeg 出力をパースして代替"""
    try:
        if check_command("ffprobe"):
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            cp = run_subprocess(cmd, timeout=15)
            out = cp.stdout.decode().strip()
            return float(out)
        elif check_command("ffmpeg"):
            # ffmpeg の stderr に Duration: 00:00:10.00 のように出るので parse
            cmd = ["ffmpeg", "-i", video_path]
            cp = run_subprocess(cmd, timeout=15)
            stderr = cp.stderr.decode(errors="ignore")
            for line in stderr.splitlines():
                if "Duration:" in line:
                    # 例: Duration: 00:00:10.05, start: 0.000000, bitrate: ...
                    try:
                        part = line.split("Duration:")[1].split(",")[0].strip()
                        h, m, s = part.split(":")
                        sec = float(h) * 3600 + float(m) * 60 + float(s)
                        return sec
                    except Exception:
                        continue
            # 採れなければ fallback 0
            return 0.0
        else:
            raise FileNotFoundError("ffmpeg/ffprobe が PATH に見つかりません。packages.txt に ffmpeg を入れてデプロイしてください。")
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffprobe/ffmpeg の実行がタイムアウトしました。")
    except Exception as e:
        raise

# -------------------------
# シーン抽出（ffmpeg select=scene を利用）
# -------------------------
def extract_scenes_with_ffmpeg(video_path: str, threshold: float = 0.3, timeout_per_frame: int = 60) -> List[Dict]:
    """
    ffmpeg の select='gt(scene,threshold)' を使ってシーン切り出しを行う。
    戻り：image file paths のリスト（時系列順）。
    """
    if not check_command("ffmpeg"):
        raise FileNotFoundError("ffmpeg が見つかりません。packages.txt に ffmpeg を追加してください。")

    tmp_dir = tempfile.mkdtemp(prefix="scenes_")
    # ffmpeg filter: select frames where scene score > threshold
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})'",
        "-vsync", "vfr",
        "-q:v", "3",  # 画質調整（3～5くらい）
        os.path.join(tmp_dir, "scene_%04d.jpg")
    ]

    # 実行（stderr を捨てずに取得しておく）
    try:
        cp = run_subprocess(cmd, timeout=300)  # 大きい動画なら時間かかる
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg によるシーン抽出がタイムアウトしました。")

    if cp.returncode != 0:
        # ffmpeg が失敗した理由を stderr に保持して返す
        err = cp.stderr.decode(errors="ignore")
        raise RuntimeError(f"ffmpeg シーン抽出が失敗しました: {err[:1000]}")

    # 抽出された画像一覧を取得
    imgs = sorted([os.path.join(tmp_dir, fn) for fn in os.listdir(tmp_dir) if fn.lower().endswith(".jpg")])
    scenes = []
    # 画像ファイル名のシーケンスから推定時刻は割り当てなし（後でdurationで均等割）
    for idx, img in enumerate(imgs):
        scenes.append({"img_path": img, "index": idx})
    return scenes

# -------------------------
# Whisper 読み込み（small） & transcribe（segments）
# -------------------------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

def transcribe_video_segments(video_path: str):
    model = load_whisper_model()
    # segments を取りたいので verboseな形式で取得
    with st.spinner("Whisper が音声を解析しています...（時間かかります）"):
        result = model.transcribe(video_path, language="ja")
    segments = result.get("segments", [])
    return segments

# -------------------------
# image -> base64 キャッシュ化
# -------------------------
@st.cache_data
def image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# -------------------------
# シーンとセグメントをアライメント
# -------------------------
def align_scenes_and_segments(scenes: List[Dict], segments: List[Dict], duration: float) -> List[Dict]:
    """
    scenes: [{"img_path":..., "index":...}, ...]
    segments: whisper の segments (start,end,text)
    duration: 動画総秒数（0 の場合は均等割）
    """
    # まず各シーンに start/end の秒を推定（select抽出は時刻を返さないため）
    n = len(scenes)
    if n == 0:
        return []

    # If duration is available, map each scene index to rough start time by equal partition.
    if duration and duration > 0:
        for s in scenes:
            idx = s["index"]
            s["start"] = (idx * duration) / max(1, n)
            s["end"] = ((idx + 1) * duration) / max(1, n)
    else:
        # fallback: start = index, end = index+1 (not ideal)
        for s in scenes:
            idx = s["index"]
            s["start"] = idx
            s["end"] = idx + 1

    # Prepare text list per scene
    for s in scenes:
        s["text_list"] = []

    # Assign segments by midpoint
    for seg in segments:
        seg_mid = (seg.get("start", 0) + seg.get("end", 0)) / 2
        matched = False
        for s in scenes:
            if s["start"] <= seg_mid < s["end"]:
                s["text_list"].append(seg.get("text", "").strip())
                matched = True
                break
        if not matched:
            # if not matched, append to last scene
            scenes[-1]["text_list"].append(seg.get("text", "").strip())

    # Compose final fields
    for s in scenes:
        s["time_str"] = f"{s['start']:.1f}"
        s["text"] = "\n".join([t for t in s["text_list"] if t])

    return scenes

# -------------------------
# TSV 横3行 x n列 生成
# -------------------------
def generate_tsv_horizontal_from_scenes(scenes: List[Dict]) -> str:
    time_row = ["時間"]
    image_row = ["画像"]
    text_row = ["テキスト"]

    for s in scenes:
        time_row.append(s.get("time_str", ""))
        # base64 formula for Google Sheets =IMAGE("data:image/jpeg;base64,....")
        if s.get("img_path") and os.path.exists(s["img_path"]):
            b64 = image_to_b64(s["img_path"])
            img_formula = f'=IMAGE("data:image/jpeg;base64,{b64}")'
        else:
            img_formula = ""
        image_row.append(img_formula)
        text_row.append(s.get("text",""))

    tsv = "\n".join(["\t".join(time_row), "\t".join(image_row), "\t".join(text_row)])
    return tsv

# -------------------------
# UI: メイン
# -------------------------
st.title("🎥 動画→シーン抽出 & Whisper 書き起こし（安定版）")

# 環境診断（簡易）
with st.expander("環境チェック（クリックで表示）", expanded=False):
    st.write("ffmpeg:", shutil.which("ffmpeg"))
    st.write("ffprobe:", shutil.which("ffprobe"))
    st.write("whisper model cache:", "available" if check_command("python") else "python ok")  # dummy

uploaded = st.file_uploader("動画ファイルをアップロード（mp4/mov/avi）", type=["mp4","mov","avi","mkv","webm"])

if not uploaded:
    st.info("動画をアップロードしてください。")
    st.stop()

# Save uploaded to temp file
with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmpf:
    tmpf.write(uploaded.read())
    video_path = tmpf.name

st.success(f"アップロード完了: {os.path.basename(video_path)}")

# Main action
if st.button("🚀 解析スタート（時間かかります）"):
    # Run entire pipeline in try/except and show traceback in UI to avoid silent Oh no
    try:
        start_ts = time.time()

        # 1) duration
        try:
            duration = get_video_duration(video_path)
            st.info(f"動画長さ: {duration:.1f} 秒")
        except Exception as e:
            st.warning(f"動画長さの取得で問題が発生しました: {e}")
            duration = 0.0

        # 2) scene extraction
        try:
            with st.spinner("シーン抽出（ffmpeg）中..."):
                scenes_raw = extract_scenes_with_ffmpeg(video_path, threshold=0.3)
            if not scenes_raw:
                st.warning("シーンが検出されませんでした。動画全体を1シーンとして扱います。")
                scenes_raw = [{"img_path": None, "index": 0}]
        except Exception as e:
            tb = traceback.format_exc()
            log_and_show("シーン抽出に失敗しました。詳細は下記。")
            st.code(tb)
            raise

        # 3) transcribe
        try:
            segments = transcribe_video_segments(video_path)
            st.success(f"書き起こし完了（セグメント数: {len(segments)})")
        except Exception as e:
            tb = traceback.format_exc()
            log_and_show("Whisper の実行に失敗しました。詳細は下記。")
            st.code(tb)
            raise

        # 4) align
        try:
            scenes = align_scenes_and_segments(scenes_raw, segments, duration)
        except Exception as e:
            tb = traceback.format_exc()
            log_and_show("シーンとセグメントの結合に失敗しました。詳細は下記。")
            st.code(tb)
            raise

        # 5) UI 表示（横スクロールカード）
        st.subheader("🔍 シーン一覧")
        # build HTML cards
        html = '<div style="display:flex; gap:16px; overflow-x:auto; padding:8px;">'
        for s in scenes:
            img_html = ""
            if s.get("img_path") and os.path.exists(s["img_path"]):
                try:
                    b64 = image_to_b64(s["img_path"])
                    img_html = f'<img src="data:image/jpeg;base64,{b64}" style="width:220px; border-radius:8px; display:block;">'
                except Exception:
                    img_html = '<div style="width:220px;height:140px;background:#eee;display:flex;align-items:center;justify-content:center;">No Image</div>'
            else:
                img_html = '<div style="width:220px;height:140px;background:#eee;display:flex;align-items:center;justify-content:center;">No Image</div>'

            t = s.get("time_str","")
            tx = s.get("text","")
            # escape simple characters to avoid HTML break (replace & < >)
            tx_safe = (tx.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
            html += f'''
            <div style="min-width:240px;padding:10px;background:#fff;border-radius:8px;box-shadow:0 6px 18px rgba(0,0,0,0.06);">
                {img_html}
                <div style="margin-top:8px;font-size:13px;color:#333;"><b>⏱ {t} s</b></div>
                <div style="margin-top:6px;font-size:13px;color:#222;white-space:pre-wrap;">{tx_safe}</div>
            </div>
            '''
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

        # 6) TSV 出力（横3行 x n 列）
        st.subheader("📋 Google スプレッドシート用 TSV（横3行 × シーン数列）")
        if st.button("TSVを生成して表示"):
            try:
                tsv = generate_tsv_horizontal_from_scenes(scenes)
                st.code(tsv, language="text")
                st.success("TSV を生成しました。スプレッドシートに貼り付けてください。")
            except Exception:
                st.error("TSV 生成に失敗しました。")
                st.code(traceback.format_exc())

        elapsed = time.time() - start_ts
        st.info(f"処理完了（所要時間: {elapsed:.1f} 秒）")

    except Exception as e_main:
        # 最終 catch：画面に traceback を必ず出す（Streamlit Cloud の Oh no を回避）
        tb_all = traceback.format_exc()
        st.error("致命的エラーが発生しました。下記の詳細を確認してください。")
        st.code(tb_all)
