import streamlit as st
from scripts.data_loader import load_data, preprocess_data
from scripts.model import train_random_forest, train_deep_learning, evaluate_model
from scripts.real_time_monitoring import real_time_monitoring, process_bag_file, process_recorded_video
from scripts.config import Config
from scripts.utils import check_gpu
import joblib
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import tensorflow as tf

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the model files
prototxt_path = os.path.join(current_dir, "models", "MobileNetSSD_deploy.prototxt")
model_path = os.path.join(current_dir, "models", "MobileNetSSD_deploy.caffemodel")

def handle_class_imbalance(X, y):
    if X is None or y is None:
        st.error("Cannot handle class imbalance: X or y is None")
        return None, None
    
    st.write("Handling class imbalance using SMOTE...")
    smote = SMOTE(random_state=42)
    try:
        X_resampled, y_resampled = smote.fit_resample(X, y)
        st.write(f"Original shape: {X.shape}, Resampled shape: {X_resampled.shape}")
        return X_resampled, y_resampled
    except Exception as e:
        st.error(f"Error in SMOTE: {str(e)}")
        return None, None
def main():
    st.title("Fall Detection with Camera's")

    check_gpu()

    try:
        # Load the MobileNet-SSD model
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        st.success("MobileNet-SSD model loaded successfully.")
    except cv2.error as e:
        st.error(f"Error loading MobileNet-SSD model: {str(e)}")
        st.error("Please make sure the model files are in the correct location.")
        return

    # Load and preprocess data
    data = load_data()
    if data is None:
        st.error("Failed to load data. Please check your data files and paths.")
        return

    X_train, X_test, y_train, y_test = preprocess_data(data)
    if X_train is None or y_train is None:
        st.error("Failed to preprocess data. Please check the preprocessing step.")
        return

    # Handle class imbalance
    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)
    if X_train_resampled is None or y_train_resampled is None:
        st.error("Failed to handle class imbalance. Using original data.")
        X_train_resampled, y_train_resampled = X_train, y_train

    # Train models
    rf_model = train_random_forest(X_train_resampled, y_train_resampled)
    dl_model = train_deep_learning(X_train_resampled, y_train_resampled)

    # Evaluate models
    st.write("Model Performance:")
    evaluate_model(rf_model, X_test, y_test, "Random Forest")
    evaluate_model(dl_model, X_test, y_test, "Deep Learning")

    # Save models
    joblib.dump(rf_model, Config.RF_MODEL_PATH)
    dl_model.save(Config.DL_MODEL_PATH, save_format='keras')

    # Input selection
    input_option = st.radio("Choose input source:", ("Live Camera", "Bag File", "Recorded Video"))

    if input_option == "Bag File":
        bag_file = st.file_uploader("Upload .bag file", type="bag")
        if bag_file:
            with open("temp.bag", "wb") as f:
                f.write(bag_file.getbuffer())
            bag_file_path = "temp.bag"
        else:
            st.warning("Please upload a .bag file")
            return
    elif input_option == "Recorded Video":
        video_file = st.file_uploader("Upload recorded video", type=["mp4", "avi"])
        if video_file:
            with open("temp_video.avi", "wb") as f:
                f.write(video_file.getbuffer())
            video_file_path = "temp_video.avi"
        else:
            st.warning("Please upload a video file")
            return
    else:
        bag_file_path = None
        video_file_path = None

    if st.button('Start Monitoring/Processing'):
        try:
            if input_option == "Live Camera":
                real_time_monitoring(rf_model, dl_model, net)
            elif input_option == "Bag File":
                process_bag_file(bag_file_path, rf_model, dl_model, net)
            else:
                process_recorded_video(video_file_path, rf_model, dl_model, net)
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")

    # Clean up temporary files
    if os.path.exists("temp.bag"):
        os.remove("temp.bag")
    if os.path.exists("temp_video.avi"):
        os.remove("temp_video.avi")

if __name__ == "__main__":
    main()