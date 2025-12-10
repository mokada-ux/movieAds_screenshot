import streamlit as st
import os
import base64
import html
from PIL import Image
import pytesseract

# ==============================
# ページ設定
# ==============================
st.set_page_config(page_title="MovieAds Screenshot Analyzer", layout="wide")

# ==============================
# CSS for UI
# ==============================
st.markdown("""
<style>
body {
    background-color: #f8f9fc;
}
.scene-container {
    display: flex;
    overflow-x: auto;
    gap: 20px;
    padding: 20px;
    white-space: nowrap;
}
.scene-card {
    background: #ffffff;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    padding: 16px;
    min-width: 280px;
    max-width: 320px;
    display: inline-block;
    vertical-align: top;
}
.scene-img {
    width: 100%;
    border-radius: 10px;
    margin-bottom: 12px;
}
.scene-text {
    background: #f1f3f8;
    padding: 12px;
    border-radius: 8px;
    font-size: 14px;
    line-height: 1.5;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# FILE UPLOAD
# ==============================
st.title("📸 MovieAds Screenshot Analyzer")

uploaded_files = st.file_uploader(
    "動画から切り出したスクショ画像をアップロードしてください（複数可）",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

# ==============================
# PROCESS & DISPLAY
# ==============================
st.subheader("📷 解析結果（横スクロールできます）")

html_out = '<div class="scene-container">'

for file in uploaded_files:
    img = Image.open(file)

    # base64 エンコード
    b64 = base64.b64encode(file.getvalue()).decode()

    # OCR → HTML エスケープ
    try:
        raw_text = pytesseract.image_to_string(img, lang="jpn+eng")
        extracted_text = html.escape(raw_text)  # ← これが壊れ対策の本命！！！
    except:
        extracted_text = "OCR failed"

    # 1枚分のカード HTML
    html_out += f"""
    <div class="scene-card">
        <img src="data:image/jpeg;base64,{b64}" class="scene-img"/>
        <div class="scene-text">{extracted_text}</div>
    </div>
    """

html_out += "</div>"

# 表示
st.markdown(html_out, unsafe_allow_html=True)
