import streamlit as st
import cv2
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# 1. ページ設定
st.set_page_config(layout="wide", page_title="2値化ツール")

# デザイン調整：タイトル切れ防止とレイアウトの安定化
st.markdown("""
    <style>
    /* 全体の余白調整 */
    .block-container { padding-top: 3.5rem; padding-bottom: 1rem; }
    
    /* タイトルのズレ・切れ防止 */
    .main-title { 
        font-size: 2.2rem; 
        font-weight: bold; 
        margin-bottom: 2rem; 
        color: #31333F; 
        line-height: 1.2;
    }
    
    /* 下部のキャプション（タイトル）行を左右完全に揃えるスタイル */
    .bottom-cap-row {
        display: flex;
        align-items: center; /* 縦方向の中央揃え */
        gap: 15px;
        min-height: 45px; /* 高さを固定 */
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .img-cap {
        font-weight: bold;
        font-size: 1.1rem;
        color: #31333F;
    }

    /* 復活：取得した色を背景にする動的カラーチップ */
    .pixel-chip {
        padding: 5px 15px;
        border-radius: 20px; /* 丸角 */
        font-weight: bold;
        font-size: 1rem;
        border: 1px solid #ddd;
        box-shadow: 1px 1px 4px rgba(0,0,0,0.1);
        display: inline-flex;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">2値化インタラクティブツール</div>', unsafe_allow_html=True)

# --- サイドバー ---
with st.sidebar:
    st.markdown("###  ? Developer\n**Takeyoshi Noriyuki**\n[(c) Official Site](https://www.hinode.ed.jp/share/takeyoshi/)")
    st.divider()
    st.info("ヒストグラムをクリックして調整してください。左の画像をクリックすると輝度が表示されます。")

uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    # 画像読み込み
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # 画像情報のメトリクス
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Width", f"{width}px")
    m2.metric("Height", f"{height}px")
    m3.metric("Channels", f"{img_array.shape[2]}")
    m4.metric("Size", f"{uploaded_file.size/1024:.1f}KB")

    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    display_gray_pil = Image.fromarray(gray_img)

    # 1. ヒストグラムデータの作成
    hist_values = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()
    hist_df = pd.DataFrame({"輝度": np.arange(256), "頻度": hist_values})

    if "threshold" not in st.session_state:
        st.session_state.threshold = 127

    # 2. 統合グラフ
    click_selection = alt.selection_point(name="pts", encodings=['x'], on='click', nearest=True, value=st.session_state.threshold, clear=False)

    bar_chart = alt.Chart(hist_df).mark_bar(color='black', binSpacing=0).encode(
        x=alt.X('輝度:Q', scale=alt.Scale(domain=[0, 255], padding=0), axis=alt.Axis(title="輝度 (0-255)")),
        y=alt.Y('頻度:Q', axis=alt.Axis(title="画素数")),
        tooltip=alt.value(None)
    ).properties(height=180)

    rule = alt.Chart(hist_df).mark_rule(color='red', size=2).encode(x='輝度:Q').transform_filter(click_selection)

    th = st.session_state.threshold
    band_data = pd.DataFrame([
        {"start": 0, "end": th, "mid": th/2, "color": "white", "label": "← ここまで白"},
        {"start": th, "end": 255, "mid": (th+255)/2, "color": "black", "label": "ここから黒 →"}
    ])

    band_rect = alt.Chart(band_data).mark_rect(stroke="black", strokeWidth=1).encode(
        x=alt.X('start:Q', scale=alt.Scale(domain=[0, 255], padding=0), axis=None),
        x2='end:Q',
        color=alt.Color('color:N', scale=None),
        tooltip=alt.value(None)
    ).properties(height=35)

    band_text = alt.Chart(band_data).mark_text(baseline='middle', fontWeight='bold').encode(
        x=alt.X('mid:Q'),
        text='label:N',
        color=alt.condition(alt.datum.color == 'white', alt.value('black'), alt.value('white')),
        tooltip=alt.value(None)
    )

    combined_chart = alt.vconcat(
        alt.layer(bar_chart, rule).add_params(click_selection),
        alt.layer(band_rect, band_text)
    ).resolve_scale(x='shared').configure_view(strokeWidth=0).configure_concat(spacing=5)

    select_event = st.altair_chart(combined_chart, on_select="rerun", use_container_width=True, key="main_histogram")

    if select_event and "selection" in select_event:
        sel = select_event["selection"]
        pts = sel.get("pts")
        if pts and len(pts) > 0:
            new_th = int(pts[0]["輝度"])
            if new_th != st.session_state.threshold:
                st.session_state.threshold = new_th
                st.rerun()

    # 3. 2値化処理
    _, binary_img = cv2.threshold(gray_img, st.session_state.threshold, 255, cv2.THRESH_BINARY)
    binary_pil = Image.fromarray(binary_img)

    st.divider()

    # --- 画像表示エリア ---
    col1, col2 = st.columns(2)
    display_width = 480 

    with col1:
        # 左側画像を先に表示（クリック可能）
        coords = streamlit_image_coordinates(display_gray_pil, width=display_width, key="image_left")
        
        # 左右共通の「下部キャプション行（高さ固定）」を作成
        cap_html = '<div class="bottom-cap-row"><div class="img-cap">元の画像 (Gray)</div>'
        
        # クリック座標がある場合はチップを表示
        if coords:
            scale = width / display_width
            x, y = int(coords["x"] * scale), int(coords["y"] * scale)
            if 0 <= x < width and 0 <= y < height:
                v = gray_img[y, x]
                # 輝度に応じて文字色を反転（視認性確保）
                txt_color = "white" if v < 128 else "black"
                # 復活：丸角の動的カラーチップ
                cap_html += f'<div class="pixel-chip" style="background-color: rgb({v},{v},{v}); color: {txt_color}; border: 1px solid #ccc;">'
                cap_html += f'輝度: {v} <span style="font-size:0.8rem; opacity:0.8; margin-left:8px;">({x}, {y})</span></div>'
        else:
            # ガイドメッセージ（矢印は使いません）
            cap_html += '<span style="color:#888; font-size:0.9rem;">  画像をクリックして輝度を取得</span>'
            
        cap_html += '</div>'
        
        st.markdown(cap_html, unsafe_allow_html=True)

    with col2:
        # 右側もstreamlit_image_coordinatesを使ってサイズを完全一致させる
        streamlit_image_coordinates(binary_pil, width=display_width, key="image_right")
        # 右側も同じクラスを使ってキャプションを下に揃える
        st.markdown(f'<div class="bottom-cap-row"><div class="img-cap">2値化結果 (しきい値: {st.session_state.threshold})</div></div>', unsafe_allow_html=True)

else:
    st.info("画像をアップロードしてください。")
