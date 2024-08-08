import streamlit as st  # Import the necessary libraries
from scripts.data_loader import load_data, preprocess_data  # Import the necessary functions
from scripts.model import train_random_forest, train_deep_learning, evaluate_model  # Import the necessary functions
from scripts.real_time_monitoring import real_time_monitoring, process_bag_file, process_recorded_video # Import the necessary functions
from scripts.config import Config   # Import the necessary class
from scripts.utils import check_gpu # Import the necessary functions
import joblib   # Import the necessary libraries
import os   # Import the necessary libraries
from imblearn.over_sampling import SMOTE    # Import the necessary libraries
from sklearn.model_selection import train_test_split    # Import the necessary libraries
import cv2  # Import the necessary libraries
import numpy as np  # Import the necessary libraries
import tensorflow as tf     # Import the necessary libraries

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppress TensorFlow warnings

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))    # Get the current directory
# Construct the path to the model files
prototxt_path = os.path.join(current_dir, "models", "MobileNetSSD_deploy.prototxt") # Construct the path to the prototxt file
model_path = os.path.join(current_dir, "models", "MobileNetSSD_deploy.caffemodel")  # Construct the path to the model file

def handle_class_imbalance(X, y):   # Function to handle class imbalance
    if X is None or y is None:  # If X or y is None
        st.error("Cannot handle class imbalance: X or y is None")   # Display an error message
        return None, None   # Return None
    
    st.write("Handling class imbalance using SMOTE...")         # Display a message
    smote = SMOTE(random_state=42)  # Create a SMOTE object
    try:    # Try block to handle exceptions
        X_resampled, y_resampled = smote.fit_resample(X, y)   # Resample the data
        st.write(f"Original shape: {X.shape}, Resampled shape: {X_resampled.shape}")    # Display the original and resampled shapes
        return X_resampled, y_resampled # Return the resampled data
    except Exception as e:  # Except block to handle exceptions
        st.error(f"Error in SMOTE: {str(e)}")   # Display an error message
        return None, None   # Return None
def main(): # Main function
    st.title("Fall Detection with Camera's")    # Display the title

    check_gpu() # Check if GPU is available

    try:    # Try block to handle exceptions
        # Load the MobileNet-SSD model
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)   # Load the MobileNet-SSD model
        st.success("MobileNet-SSD model loaded successfully.")  # Display a success message
    except cv2.error as e:      # Except block to handle exceptions
        st.error(f"Error loading MobileNet-SSD model: {str(e)}")    # Display an error message
        st.error("Please make sure the model files are in the correct location.")   # Display an error message
        return  # Return

    # Load and preprocess data
    data = load_data()  # Load the data
    if data is None:    # If the data is None
        st.error("Failed to load data. Please check your data files and paths.")    # Display an error message
        return  # Return

    X_train, X_test, y_train, y_test = preprocess_data(data)    # Preprocess the data
    if X_train is None or y_train is None:  # If X_train or y_train is None
        st.error("Failed to preprocess data. Please check the preprocessing step.") # Display an error message
        return  # Return

    # Handle class imbalance
    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)   # Handle class imbalance
    if X_train_resampled is None or y_train_resampled is None:  # If X_train_resampled or y_train_resampled is None
        st.error("Failed to handle class imbalance. Using original data.")  # Display an error message
        X_train_resampled, y_train_resampled = X_train, y_train  # Use the original data

    # Train models
    rf_model = train_random_forest(X_train_resampled, y_train_resampled)    # Train the Random Forest model
    dl_model = train_deep_learning(X_train_resampled, y_train_resampled)    # Train the Deep Learning model

    # Evaluate models
    st.write("Model Performance:")  # Display a message
    evaluate_model(rf_model, X_test, y_test, "Random Forest")   # Evaluate the Random Forest model
    evaluate_model(dl_model, X_test, y_test, "Deep Learning")   # Evaluate the Deep Learning model

    # Save models
    joblib.dump(rf_model, Config.RF_MODEL_PATH)  # Save the Random Forest model
    dl_model.save(Config.DL_MODEL_PATH, save_format='keras')    # Save the Deep Learning model

    # Input selection
    input_option = st.radio("Choose input source:", ("Live Camera (Not working yet)", "Bag File (not working yet)", "Recorded Video"))  # Radio button for input selection

    if input_option == "Bag File":  # If input_option is "Bag File"
        bag_file = st.file_uploader("Upload .bag file", type="bag") # File uploader for .bag file
        if bag_file:    # If bag_file is not None
            with open("temp.bag", "wb") as f:   # Open a temporary file
                f.write(bag_file.getbuffer())   # Write the contents of the file to the temporary file
            bag_file_path = "temp.bag"  # Set the bag_file_path
        else:       # If bag_file is None   
            st.warning("Please upload a .bag file") # Display a warning message
            return  # Return
    elif input_option == "Recorded Video":      # If input_option is "Recorded Video"
        video_file = st.file_uploader("Upload recorded video", type=["mp4", "avi"])  # File uploader for recorded video
        if video_file:  # If video_file is not None
            with open("temp_video.avi", "wb") as f: # Open a temporary file
                f.write(video_file.getbuffer()) # Write the contents of the file to the temporary file
            video_file_path = "temp_video.avi"  # Set the video_file_path
        else:   # If video_file is None
            st.warning("Please upload a video file")    # Display a warning message
            return  # Return
    else:   # If input_option is "Live Camera"
        bag_file_path = None    # Set bag_file_path to None
        video_file_path = None  # Set bag_file_path and video_file_path to None

    if st.button('Start Monitoring/Processing'):    # Button to start monitoring/processing
        try:    # Try block to handle exceptions
            if input_option == "Live Camera":   # If input_option is "Live Camera"
                real_time_monitoring(rf_model, dl_model, net)   # Start real-time monitoring
            elif input_option == "Bag File":    # If input_option is "Bag File"
                process_bag_file(bag_file_path, rf_model, dl_model, net)    # Process the bag file
            else:       # If input_option is "Recorded Video"
                process_recorded_video(video_file_path, rf_model, dl_model, net)    # Process the recorded video
        except Exception as e:  # Except block to handle exceptions
            st.error(f"An error occurred during processing: {str(e)}")  # Display an error message

    # Clean up temporary files
    if os.path.exists("temp.bag"):  # If "temp.bag" exists
        os.remove("temp.bag")   # Remove "temp.bag"
    if os.path.exists("temp_video.avi"):    # If "temp_video.avi" exists
        os.remove("temp_video.avi") # Remove "temp_video.avi"

if __name__ == "__main__":  # If the script is executed directly
    main()  # Call the main function