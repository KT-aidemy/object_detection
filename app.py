import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO

# タイトル
st.title("リアルタイム物体検出 (Streamlit + YOLOv8)")

# YOLOモデルの読み込み（自動ダウンロード）
@st.cache_resource
def load_model(model_name="yolov8n"):
    # 拡張子なしモデル名で自動ダウンロード＆キャッシュ
    return YOLO(model_name)

model = load_model()

# 動画ファイルアップロード
st.sidebar.header("動画アップロード")
video_file = st.sidebar.file_uploader("動画ファイルを選択", type=["mp4", "mov", "avi"])

# 推論設定
conf_thres = st.sidebar.slider("信頼度閾値", 0.0, 1.0, 0.3)
iou_thres = st.sidebar.slider("IoU閾値", 0.0, 1.0, 0.5)

if video_file:
    # 一時ファイルに保存
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    # 動画キャプチャ開始
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR->RGB変換
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 推論
        results = model(img, conf=conf_thres, iou=iou_thres)[0]

        # 描画結果取得
        annotated = results.plot()

        # Streamlit表示用にBGRに戻す
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        # 表示
        stframe.image(annotated_bgr, channels="BGR", use_column_width=True)
    cap.release()
else:
    st.info("サイドバーから動画ファイルをアップロードしてください。")

# 実行方法:
# pip install streamlit opencv-python ultralytics
# streamlit run streamlit_video_object_detection.py
```
