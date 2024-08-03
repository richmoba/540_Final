import cv2
import numpy as np
import streamlit as st
from scripts.config import Config

def extract_features(depth_frame):
    # Implement feature extraction here
    # This is a placeholder - you need to implement the actual feature extraction
    features = np.zeros((1, len(Config.FEATURES)))
    return features

def real_time_monitoring(rf_model, dl_model):
    orbbec_camera_index = None
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                orbbec_camera_index = i
                cap.release()
                break
        cap.release()

    if orbbec_camera_index is None:
        st.error("Error: Could not find the Orbbec Gemini 335 camera.")
        return

    cap = cv2.VideoCapture(orbbec_camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error("Error: Could not open video stream.")
        return

    stop_button = st.button('Stop Monitoring')
    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    frame_count = 0
    while not stop_button:
        try:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame.")
                break

            features = extract_features(frame)
            y_pred_rf = rf_model.predict(features)
            y_pred_dl = (dl_model.predict(features) > 0.5).astype("int32")

            info_placeholder.write(f"Frame {frame_count}:")
            info_placeholder.write(f"Random Forest Prediction: {y_pred_rf[0]}")
            info_placeholder.write(f"Deep Learning Prediction: {y_pred_dl[0][0]}")

            frame_resized = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, caption='Depth Frame', use_column_width=True)

            frame_count += 1

        except cv2.error as e:
            st.error(f"OpenCV error: {e}")
            break

        # Check if the stop button has been pressed
        if stop_button:
            break

    cap.release()
    cv2.destroyAllWindows()