import pyrealsense2 as rs   # Intel RealSense cross-platform open-source API
import numpy as np        # NumPy for array manipulation
import cv2              # OpenCV
import streamlit as st   # Streamlit for the web app
from scripts.config import Config   # Custom Config class for app configuration
import time            # Time module for time-related functions
import os          # OS module for file operations

# Get the current script's directory    
current_dir = os.path.dirname(os.path.abspath(__file__))                            
# Go up one level to the project root
project_root = os.path.dirname(current_dir)             
# Construct the path to the model files
prototxt_path = os.path.join(project_root, "models", "MobileNetSSD_deploy.prototxt")                
model_path = os.path.join(project_root, "models", "MobileNetSSD_deploy.caffemodel")              

# Load the MobileNet-SSD model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)               

# Global variables for tracking
prev_center = None  # Previous center of the detected person
prev_time = None    # Previous time of detection
prev_box = None    # Previous bounding box coordinates
prev_velocity = 0   # Previous velocity of the person
prev_features = None    # Previous extracted features

class OrbbecViewer:
    def __init__(self):     
        self.pipeline = rs.pipeline()   # Create a pipeline
        self.config = rs.config()   # Create a configuration object
        self.recording = False  # Recording flag
        self.out = None      # Video writer object

    def start(self):        
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # Enable depth stream
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)    # Enable color stream
        self.pipeline.start(self.config)    # Start the pipeline
        self.align = rs.align(rs.stream.color)  # Create an align object

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()    # Wait for frames
        aligned_frames = self.align.process(frames) # Align the frames
        depth_frame = aligned_frames.get_depth_frame()  # Get aligned depth frame
        color_frame = aligned_frames.get_color_frame()  # Get aligned color frame
        return depth_frame, color_frame # Return both frames

    def start_recording(self, filename):            
        fourcc = cv2.VideoWriter_fourcc(*'XVID')    # Define the codec  
        self.out = cv2.VideoWriter(filename, fourcc, 30.0, (1280, 480)) # Create the VideoWriter object
        self.recording = True   # Set the recording flag

    def stop_recording(self):   
        if self.out:    # If the VideoWriter object exists
            self.out.release()  # Release the object
        self.recording = False  # Reset the recording flag

    def stop(self):         
        self.pipeline.stop()    # Stop the pipeline


def extract_features(frame):    
    global prev_features, prev_time     # Access the global variables
    
    (h, w) = frame.shape[:2]    # Get the frame dimensions
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)    # Create a blob

    net.setInput(blob)  # Set the input to the network
    detections = net.forward()  # Perform a forward pass

    person_detected = False # Flag to track person detection
    for i in range(detections.shape[2]):    # Iterate over the detections
        confidence = detections[0, 0, i, 2]   # Get the confidence score
        if confidence > 0.5:  # Minimum confidence threshold
            idx = int(detections[0, 0, i, 1])   # Get the class ID
            if idx == 15:  # Class ID for person
                person_detected = True  # Person detected
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) # Bounding box coordinates
                (startX, startY, endX, endY) = box.astype("int")    # Convert to integers
                
                height = max(endY - startY, 1)  # Ensure non-zero height
                width = max(endX - startX, 1)   # Ensure non-zero width
                
                height_width_ratio = height / width # Aspect ratio
                bounding_box_occupancy = (height * width) / (h * w) # Occupancy ratio
                
                major_minor_ratio = max(height, width) / min(height, width) # Major/minor axis ratio
                
                roi = frame[startY:endY, startX:endX]   # Region of interest
                if roi.size > 0:    # If the ROI is not empty
                    max_std_xz = np.std(roi)    # Maximum standard deviation
                else:   # If the ROI is empty
                    max_std_xz = 0  # Set to 0
                
                hh_max_ratio = height / h   # Height ratio
                h_value = height    # Height value
                d_value = width    # Width value
                
                if roi.size > 0:    # If the ROI is not empty
                    p40 = np.percentile(roi, 40)        # 40th percentile
                else:   # If the ROI is empty
                    p40 = 0   # Set to 0

                features = [    # Extracted features
                    height_width_ratio,     # Aspect ratio
                    major_minor_ratio,    # Major/minor axis ratio
                    bounding_box_occupancy,   # Occupancy ratio
                    max_std_xz,   # Maximum standard deviation
                    hh_max_ratio,   # Height ratio
                    h_value,    # Height value
                    d_value,    # Width value
                    p40   # 40th percentile
                ]
                
                prev_features = features    # Update previous features
                prev_time = cv2.getTickCount() / cv2.getTickFrequency()   # Update previous time
                
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)    # Draw the bounding box
                break   # Break from the loop

    if not person_detected:   # If no person is detected
        features = prev_features if prev_features is not None else [1, 1, 0, 0, 0.5, h/2, w/2, 128] # Use previous features or default values

    return np.array([features])  # Return as 2D array for model input       
def real_time_monitoring(rf_model, dl_model, net):              
    st.write("Starting real-time monitoring...")    # Display a message

    viewer = OrbbecViewer()   # Create an OrbbecViewer object
    viewer.start()  # Start the viewer

    frame_placeholder = st.empty()  # Create an empty placeholder for the frame
    info_placeholder = st.empty()   # Create an empty placeholder for information

    col1, col2, col3 = st.columns(3)    # Create 3 columns
    stop_button = col1.button('Stop Monitoring')    # Button to stop monitoring
    record_button = col2.button('Start Recording')  # Button to start recording
    stop_record_button = col3.button('Stop Recording')  # Button to stop recording

    try:    # Try block to handle exceptions
        while not stop_button:  # While the stop button is not pressed
            depth_frame, color_frame = viewer.get_frames()  # Get the frames

            if not depth_frame or not color_frame:  # If frames are not available 
                continue    # Skip the rest of the loop

            depth_image = np.asanyarray(depth_frame.get_data()) # Convert depth frame to NumPy array
            color_image = np.asanyarray(color_frame.get_data()) # Convert color frame to NumPy array

            features = extract_features(color_image)    # Extract features from the color image

            y_pred_rf = rf_model.predict(features)  # Random Forest prediction
            y_pred_dl = (dl_model.predict(features) > 0.5).astype("int32")  # Deep Learning prediction

            info_placeholder.write(f"Random Forest Prediction: {y_pred_rf[0]}") # Display RF prediction
            info_placeholder.write(f"Deep Learning Prediction: {y_pred_dl[0][0]}")  # Display DL prediction

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)  # Apply colormap to depth image
            images = np.hstack((color_image, depth_colormap))       # Combine color and depth images

            frame_placeholder.image(images, caption='Color and Depth Frames', channels="BGR")   # Display the images

            if record_button:   # If the record button is pressed
                if not viewer.recording:    # If not already recording
                    filename = f"recording_{time.strftime('%Y%m%d-%H%M%S')}.avi"    # Create a filename
                    viewer.start_recording(filename)    # Start recording
                    st.success(f"Started recording: {filename}")    # Display a success message
                record_button = False   # Reset the button

            if stop_record_button:  # If the stop record button is pressed
                if viewer.recording:    # If currently recording
                    viewer.stop_recording()   # Stop recording
                    st.success("Stopped recording")   # Display a success message
                stop_record_button = False  # Reset the button

            if viewer.recording:    # If currently recording
                viewer.out.write(images)    # Write the images to the video

            stop_button = st.button('Stop Monitoring')  # Update the stop button

    except Exception as e:  # Catch any exceptions
        st.error(f"An error occurred: {str(e)}")    # Display an error message

    finally:    # Finally block to ensure cleanup
        viewer.stop()   # Stop the viewer
        if viewer.recording:    # If still recording
            viewer.stop_recording() # Stop recording

def process_recorded_video(video_path, rf_model, dl_model, net):                
    st.write(f"Processing recorded video: {video_path}")    # Display the video path

    cap = cv2.VideoCapture(video_path)  # Open the video file      
    frame_placeholder = st.empty()  # Create an empty placeholder for the frame
    info_placeholder = st.empty()   # Create an empty placeholder for information

    previous_state = "Standing"   # Previous state
    fall_detected = False   # Fall detected flag
    
    frame_count = 0  # Frame count
    fall_frames = 0     # Number of fall frames
    standing_frames = 0 # Number of standing frames
    transition_count = 0    # Number of transitions

    fall_start_frame = None # Frame number where fall starts
    potential_fall_frames = 0   # Number of potential fall frames
    fall_confidence_threshold = 0.5  # Lowered threshold for fall detection
    fall_duration_threshold = 5  # Minimum number of frames to confirm a fall

    fall_window = []    # Window to store fall likelihood values
    window_size = 15  # Consider last 15 frames for fall detection

    while cap.isOpened():   # While the video is open
        ret, frame = cap.read() # Read a frame
        if not ret:    # If the frame is not valid
            break   # Break from the loop

        frame_count += 1    # Increment the frame count
        features = extract_features(frame)  # Extract features from the frame

        y_pred_rf = rf_model.predict(features)[0]   # Random Forest prediction
        y_pred_dl = dl_model.predict(features)[0][0]  # Assuming it returns a probability

        fall_likelihood = (y_pred_rf + y_pred_dl) / 2  # Average of both model predictions
        
        height_width_ratio, bounding_box_occupancy, vertical_position, _, _, _, velocity, acceleration = features[0]    # Extract features
               
        fall_window.append(fall_likelihood) # Append the fall likelihood to the window
        if len(fall_window) > window_size:  # If the window size exceeds the limit
            fall_window.pop(0)  # Remove the oldest value

        avg_fall_likelihood = sum(fall_window) / len(fall_window)       # Average fall likelihood

        if height_width_ratio < 0.8 or vertical_position > 0.7 or (abs(velocity) > 50 and abs(acceleration) > 100): # Fall-like behavior
            potential_fall_frames += 1  # Increment potential fall frames
        else:   # Not fall-like behavior
            potential_fall_frames = max(0, potential_fall_frames - 1)   # Reset potential fall frames

        if avg_fall_likelihood > fall_confidence_threshold or potential_fall_frames > fall_duration_threshold:  # Fall detected
            current_state = "Fallen"    # Set the current state to fallen
            fall_frames += 1    # Increment the fall frames
            if previous_state != "Fallen":  # If the previous state was not fallen
                fall_detected = True    # Set the fall detected flag
                transition_count += 1   # Increment the transition count
                fall_start_frame = frame_count  # Set the fall start frame
        else:   # Not fallen
            current_state = "Standing"  # Set the current state to standing
            standing_frames += 1    # Increment the standing frames

        if fall_detected:   # If fall detected
            info_placeholder.error(f"FALL DETECTED at frame {fall_start_frame}!")   # Display a fall detected message
            fall_detected = False  # Reset for next potential fall      
        
        info_placeholder.write(f"Current State: {current_state}")   # Display the current state
        info_placeholder.write(f"Fall Likelihood: {avg_fall_likelihood:.2f}")   # Display the fall likelihood
        info_placeholder.write(f"Potential Fall Frames: {potential_fall_frames}")   # Display potential fall frames
        info_placeholder.write(f"Features: {features[0]}")  # Display the extracted features

        frame_placeholder.image(frame, caption='Recorded Video', channels="BGR")    # Display the frame

        previous_state = current_state  # Update the previous state

    cap.release()   # Release the video capture

    # Provide summary
    st.write("Video Processing Complete. Summary:")   # Display a summary
    st.write(f"Total frames processed: {frame_count}")  # Display the total frames
    st.write(f"Frames classified as standing: {standing_frames} ({standing_frames/frame_count*100:.2f}%)")  # Display standing frames 
    st.write(f"Frames classified as fallen: {fall_frames} ({fall_frames/frame_count*100:.2f}%)")        # Display fallen frames
    st.write(f"Number of transitions detected: {transition_count}") # Display the number of transitions

    if fall_frames > 0: # If falls detected
        st.write("The video contains fall events.") # Display fall events message
        if transition_count > 1:    # If multiple transitions
            st.write("Multiple fall/recovery events were detected.")    # Display multiple events message
        elif fall_frames == frame_count:    # If all frames are falls
            st.write("The entire video shows a fallen state.")  # Display all fallen message
        else:   # Single fall
            st.write(f"A fall event was detected starting at frame {fall_start_frame}.")    # Display single fall message
    else:   # No falls detected
        st.write("No definitive falls were detected in this video.")    # Display no falls message
        if potential_fall_frames > 0:   # If potential falls detected
            st.write(f"However, {potential_fall_frames} frames showed potential fall-like behavior.")   # Display potential falls message
        if standing_frames == frame_count:  # If all frames are standing
            st.write("The video mostly shows normal standing/walking behavior.")    # Display all standing message
        else:   # Some unstable movements
            st.write("The video may contain some unstable movements, but no clear falls.")  

    if transition_count > 0:    # If transitions detected
        st.write(f"On average, a state change (fall or recovery) occurred every {frame_count/transition_count:.2f} frames.")    # Display average transition message

def process_bag_file(bag_file_path, rf_model, dl_model, net):                 
    st.write(f"Processing bag file: {bag_file_path}")   # Display the bag file path

    pipeline = rs.pipeline()    # Create a pipeline
    config = rs.config()    # Create a configuration object

    rs.config.enable_device_from_file(config, bag_file_path)    # Enable device from file

    config.enable_stream(rs.stream.depth, rs.format.z16, 30)    # Enable depth stream
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)   # Enable color stream

    pipeline.start(config)  # Start the pipeline

    align_to = rs.stream.color  # Align to color stream
    align = rs.align(align_to)  # Create an align object

    frame_placeholder = st.empty()  # Create an empty placeholder for the frame
    info_placeholder = st.empty()   # Create an empty placeholder for information

    frame_count = 0 # Frame count
    fall_frames = 0 # Number of fall frames
    standing_frames = 0 # Number of standing frames
    transition_count = 0    # Number of transitions
    previous_state = "Standing" # Previous state
    fall_detected = False   # Fall detected flag
    fall_start_frame = None # Frame number where fall starts
    potential_fall_frames = 0   # Number of potential fall frames

    try:    # Try block to handle exceptions
        while True:   # Infinite loop
            frames = pipeline.wait_for_frames() # Wait for frames
            aligned_frames = align.process(frames)  # Align the frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # Get aligned depth frame
            color_frame = aligned_frames.get_color_frame()  # Get aligned color frame

            if not aligned_depth_frame or not color_frame:  # If frames are not available
                continue    # Skip the rest of the loop

            depth_image = np.asanyarray(aligned_depth_frame.get_data()) # Convert depth frame to NumPy array
            color_image = np.asanyarray(color_frame.get_data()) # Convert color frame to NumPy array

            frame_count += 1    # Increment the frame count
            features = extract_features(color_image)    # Extract features from the color image

            y_pred_rf = rf_model.predict(features)[0]   # Random Forest prediction
            y_pred_dl = (dl_model.predict(features) > 0.5).astype("int32")[0][0]    # Deep Learning prediction

            fall_likelihood = (y_pred_rf + y_pred_dl) / 2   # Average of both model predictions

            height_width_ratio, _, vertical_position, _, _, _, velocity, acceleration = features[0]   # Extract features

            if height_width_ratio < 1.2 or vertical_position > 0.7 or (abs(velocity) > 50 and abs(acceleration) > 100): # Fall-like behavior
                potential_fall_frames += 1  # Increment potential fall frames
            else:   # Not fall-like behavior
                potential_fall_frames = max(0, potential_fall_frames - 1)   # Reset potential fall frames

            if potential_fall_frames > 5 or fall_likelihood > 0.5:  # Fall detected
                current_state = "Fallen"    # Set the current state to fallen
                fall_frames += 1    # Increment the fall frames
                if previous_state != "Fallen":  # If the previous state was not fallen
                    fall_detected = True    # Set the fall detected flag
                    transition_count += 1   # Increment the transition count
                    fall_start_frame = frame_count  # Set the fall start frame
            else:   # Not fallen
                current_state = "Standing"  # Set the current state to standing
                standing_frames += 1    # Increment the standing frames

            if fall_detected:   # If fall detected
                info_placeholder.error(f"FALL DETECTED at frame {fall_start_frame}!")   # Display a fall detected message
                fall_detected = False   # Reset for next potential fall

            info_placeholder.write(f"Current State: {current_state}")   # Display the current state
            info_placeholder.write(f"Fall Likelihood: {fall_likelihood:.2f}")   # Display the fall likelihood
            info_placeholder.write(f"Potential Fall Frames: {potential_fall_frames}")   # Display potential fall frames
            info_placeholder.write(f"Features: {features[0]}")      # Display the extracted features

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)  # Apply colormap to depth image
            images = np.hstack((color_image, depth_colormap))   # Combine color and depth images
            frame_placeholder.image(images, caption='Color and Depth Frames', channels="RGB")   # Display the images

            previous_state = current_state  # Update the previous state
            time.sleep(0.033)  # Adjust this value to change the playback speed

    except RuntimeError:    # Catch a runtime error
        st.write("Reached the end of the bag file.")    # Display a message
    except Exception as e:  # Catch any other exceptions
        st.error(f"An error occurred: {str(e)}")    # Display an error message
    finally:    # Finally block to ensure cleanup
        pipeline.stop() # Stop the pipeline

    # Provide summary   
    st.write("Bag File Processing Complete. Summary:")  # Display a summary
    st.write(f"Total frames processed: {frame_count}")  # Display the total frames
    st.write(f"Frames classified as standing: {standing_frames} ({standing_frames/frame_count*100:.2f}%)")  # Display standing frames
    st.write(f"Frames classified as fallen: {fall_frames} ({fall_frames/frame_count*100:.2f}%)")    # Display
    st.write(f"Number of transitions detected: {transition_count}") # Display the number of transitions

    if fall_frames > 0: # If falls detected
        st.write("The bag file contains fall events.")  # Display fall events message
        if transition_count > 1:    # If multiple transitions
            st.write("Multiple fall/recovery events were detected.")    # Display multiple events message
        elif fall_frames == frame_count:    # If all frames are falls
            st.write("The entire recording shows a fallen state.")  # Display all fallen message
        else:   # Single fall
            st.write(f"A fall event was detected starting at frame {fall_start_frame}.")    # Display single fall message
    else:   # No falls detected
        st.write("No definitive falls were detected in this recording.")    # Display no falls message
        if potential_fall_frames > 0:   # If potential falls detected
            st.write(f"However, {potential_fall_frames} frames showed potential fall-like behavior.")   # Display potential falls message
        if standing_frames == frame_count:  # If all frames are standing
            st.write("The recording mostly shows normal standing/walking behavior.")    # Display all standing message
        else:   # Some unstable movements
            st.write("The recording may contain some unstable movements, but no clear falls.")                      

    if transition_count > 0:    # If transitions detected
        st.write(f"On average, a state change (fall or recovery) occurred every {frame_count/transition_count:.2f} frames.")    # Display average transition message