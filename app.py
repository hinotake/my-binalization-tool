import streamlit as st
import cv2
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image

# 1. ページ設定
st.set_page_config(layout="wide", page_title="2値化インタラクティブツール")

# 余白とフォントサイズの微調整
st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 0rem; }
    h1 { font-size: 1.6rem !important; margin-bottom: 0.5rem; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("2値化インタラクティブツール")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 画像の読み込み
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # 基本情報の取得
    height, width = img_array.shape[:2]
    channels = img_array.shape[2] if len(img_array.shape) == 3 else 1
    file_size = uploaded_file.size / 1024  # KB単位
    
    # --- 【画像情報を一番上に配置】 ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Width", f"{width} px")
    m2.metric("Height", f"{height} px")
    m3.metric("Channels", f"{channels}")
    m4.metric("Size", f"{file_size:.1f} KB")

    if len(img_array.shape) == 3:
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img_array

    # 1. ヒストグラムデータの作成
    hist_values, bin_edges = np.histogram(gray_img, bins=256, range=(0, 256))
    hist_df = pd.DataFrame({
        "輝度": bin_edges[:-1].astype(int),
        "頻度": hist_values
    })

    # 2. Altairグラフの設定
    click_selection = alt.selection_point(
        name="pts", 
        encodings=['x'], 
        on='click', 
        nearest=True, 
        value=127
    )

    # ヒストグラム：黒色
    bar = alt.Chart(hist_df).mark_bar(color='black').encode(
        x=alt.X('輝度:Q', scale=alt.Scale(domain=[0, 255]), title="しきい値選択 (0-255)"),
        y=alt.Y('頻度:Q', axis=None) 
    )

    rule = alt.Chart(hist_df).mark_rule(color='red', size=3).encode(
        x='輝度:Q'
    ).transform_filter(click_selection)

    # 【修正箇所】title=None を title="" に変更
    chart = alt.layer(bar, rule).add_params(
        click_selection
    ).properties(
        width="container", 
        height=180, 
        title=""
    )

    # グラフを表示
    select_event = st.altair_chart(chart, on_select="rerun", use_container_width=True)

    # --- しきい値取得ロジック ---
    threshold = 127 
    if select_event and "selection" in select_event:
        sel = select_event["selection"]
        points = sel.get("pts") or sel.get("point")
        if points and len(points) > 0:
            point_data = points[0]
            val_list = list(point_data.values())
            if val_list:
                try:
                    threshold = int(val_list[0])
                except (ValueError, TypeError):
                    threshold = 127

    # 3. 2値化処理
    _, binary_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)

    st.divider()
    st.markdown(f"**現在のしきい値: {threshold}**")

    col1, col2 = st.columns(2)
    with col1:
        st.image(gray_img, caption="元の画像", use_container_width=True)
    with col2:
        st.image(binary_img, caption="2値化結果", use_container_width=True)

else:
    st.info("画像をアップロードしてください。")