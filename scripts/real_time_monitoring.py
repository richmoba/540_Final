import pyrealsense2 as rs
import numpy as np
import cv2
import streamlit as st
from scripts.config import Config
import time
import os

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
prev_center = None
prev_time = None
prev_box = None
prev_velocity = 0
prev_features = None

class OrbbecViewer:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.recording = False
        self.out = None

    def start(self):
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        return depth_frame, color_frame

    def start_recording(self, filename):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(filename, fourcc, 30.0, (1280, 480))
        self.recording = True

    def stop_recording(self):
        if self.out:
            self.out.release()
        self.recording = False

    def stop(self):
        self.pipeline.stop()

# def extract_features(frame):
#     global prev_features, prev_time
    
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

#     net.setInput(blob)
#     detections = net.forward()

#     person_detected = False
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.5:  # Minimum confidence threshold
#             idx = int(detections[0, 0, i, 1])
#             if idx == 15:  # Class ID for person
#                 person_detected = True
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")
                
#                 height_width_ratio = (endY - startY) / (endX - startX)
#                 bounding_box_occupancy = ((endX - startX) * (endY - startY)) / (w * h)
#                 vertical_position = endY / h
                
#                 velocity = 0
#                 acceleration = 0
#                 current_time = cv2.getTickCount() / cv2.getTickFrequency()
#                 if prev_features is not None and prev_time is not None:
#                     time_diff = current_time - prev_time
#                     prev_center = (prev_features[0] * w, prev_features[2] * h)
#                     current_center = ((startX + endX) / 2, (startY + endY) / 2)
#                     distance = np.sqrt((current_center[0] - prev_center[0])**2 + (current_center[1] - prev_center[1])**2)
#                     velocity = distance / time_diff if time_diff > 0 else 0
#                     acceleration = (velocity - prev_features[5]) / time_diff if time_diff > 0 else 0
                
#                 features = [
#                     height_width_ratio,
#                     bounding_box_occupancy,
#                     vertical_position,
#                     velocity,
#                     acceleration,
#                     confidence,
#                     (endX - startX) / w,  # Width ratio
#                     (endY - startY) / h   # Height ratio
#                 ]
                
#                 prev_features = features
#                 prev_time = current_time
                
#                 cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
#                 break

#     if not person_detected:
#         features = prev_features if prev_features is not None else [1, 0, 0.5, 0, 0, 0, 0.5, 0.5]

#     return np.array([features])  # Return as 2D array for model input
def extract_features(frame):
    global prev_features, prev_time
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    person_detected = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Minimum confidence threshold
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # Class ID for person
                person_detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                height = max(endY - startY, 1)  # Ensure non-zero height
                width = max(endX - startX, 1)   # Ensure non-zero width
                
                height_width_ratio = height / width
                bounding_box_occupancy = (height * width) / (h * w)
                
                major_minor_ratio = max(height, width) / min(height, width)
                
                roi = frame[startY:endY, startX:endX]
                if roi.size > 0:
                    max_std_xz = np.std(roi)
                else:
                    max_std_xz = 0
                
                hh_max_ratio = height / h
                h_value = height
                d_value = width
                
                if roi.size > 0:
                    p40 = np.percentile(roi, 40)
                else:
                    p40 = 0

                features = [
                    height_width_ratio,
                    major_minor_ratio,
                    bounding_box_occupancy,
                    max_std_xz,
                    hh_max_ratio,
                    h_value,
                    d_value,
                    p40
                ]
                
                prev_features = features
                prev_time = cv2.getTickCount() / cv2.getTickFrequency()
                
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                break

    if not person_detected:
        features = prev_features if prev_features is not None else [1, 1, 0, 0, 0.5, h/2, w/2, 128]

    return np.array([features])  # Return as 2D array for model input
def real_time_monitoring(rf_model, dl_model, net):
    st.write("Starting real-time monitoring...")

    viewer = OrbbecViewer()
    viewer.start()

    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    col1, col2, col3 = st.columns(3)
    stop_button = col1.button('Stop Monitoring')
    record_button = col2.button('Start Recording')
    stop_record_button = col3.button('Stop Recording')

    try:
        while not stop_button:
            depth_frame, color_frame = viewer.get_frames()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            features = extract_features(color_image)

            y_pred_rf = rf_model.predict(features)
            y_pred_dl = (dl_model.predict(features) > 0.5).astype("int32")

            info_placeholder.write(f"Random Forest Prediction: {y_pred_rf[0]}")
            info_placeholder.write(f"Deep Learning Prediction: {y_pred_dl[0][0]}")

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))

            frame_placeholder.image(images, caption='Color and Depth Frames', channels="BGR")

            if record_button:
                if not viewer.recording:
                    filename = f"recording_{time.strftime('%Y%m%d-%H%M%S')}.avi"
                    viewer.start_recording(filename)
                    st.success(f"Started recording: {filename}")
                record_button = False

            if stop_record_button:
                if viewer.recording:
                    viewer.stop_recording()
                    st.success("Stopped recording")
                stop_record_button = False

            if viewer.recording:
                viewer.out.write(images)

            stop_button = st.button('Stop Monitoring')

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    finally:
        viewer.stop()
        if viewer.recording:
            viewer.stop_recording()

def process_recorded_video(video_path, rf_model, dl_model, net):
    st.write(f"Processing recorded video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    previous_state = "Standing"
    fall_detected = False
    
    frame_count = 0
    fall_frames = 0
    standing_frames = 0
    transition_count = 0

    fall_start_frame = None
    potential_fall_frames = 0
    fall_confidence_threshold = 0.5  # Lowered threshold for fall detection
    fall_duration_threshold = 5  # Minimum number of frames to confirm a fall

    fall_window = []
    window_size = 15  # Consider last 15 frames for fall detection

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        features = extract_features(frame)

        y_pred_rf = rf_model.predict(features)[0]
        y_pred_dl = dl_model.predict(features)[0][0]  # Assuming it returns a probability

        fall_likelihood = (y_pred_rf + y_pred_dl) / 2  # Average of both model predictions
        
        height_width_ratio, bounding_box_occupancy, vertical_position, _, _, _, velocity, acceleration = features[0]
               
        fall_window.append(fall_likelihood)
        if len(fall_window) > window_size:
            fall_window.pop(0)

        avg_fall_likelihood = sum(fall_window) / len(fall_window)

        if height_width_ratio < 0.8 or vertical_position > 0.7 or (abs(velocity) > 50 and abs(acceleration) > 100):
            potential_fall_frames += 1
        else:
            potential_fall_frames = max(0, potential_fall_frames - 1)

        if avg_fall_likelihood > fall_confidence_threshold or potential_fall_frames > fall_duration_threshold:
            current_state = "Fallen"
            fall_frames += 1
            if previous_state != "Fallen":
                fall_detected = True
                transition_count += 1
                fall_start_frame = frame_count
        else:
            current_state = "Standing"
            standing_frames += 1

        if fall_detected:
            info_placeholder.error(f"FALL DETECTED at frame {fall_start_frame}!")
            fall_detected = False  # Reset for next potential fall
        
        info_placeholder.write(f"Current State: {current_state}")
        info_placeholder.write(f"Fall Likelihood: {avg_fall_likelihood:.2f}")
        info_placeholder.write(f"Potential Fall Frames: {potential_fall_frames}")
        info_placeholder.write(f"Features: {features[0]}")

        frame_placeholder.image(frame, caption='Recorded Video', channels="BGR")

        previous_state = current_state

    cap.release()

    # Provide summary
    st.write("Video Processing Complete. Summary:")
    st.write(f"Total frames processed: {frame_count}")
    st.write(f"Frames classified as standing: {standing_frames} ({standing_frames/frame_count*100:.2f}%)")
    st.write(f"Frames classified as fallen: {fall_frames} ({fall_frames/frame_count*100:.2f}%)")
    st.write(f"Number of transitions detected: {transition_count}")

    if fall_frames > 0:
        st.write("The video contains fall events.")
        if transition_count > 1:
            st.write("Multiple fall/recovery events were detected.")
        elif fall_frames == frame_count:
            st.write("The entire video shows a fallen state.")
        else:
            st.write(f"A fall event was detected starting at frame {fall_start_frame}.")
    else:
        st.write("No definitive falls were detected in this video.")
        if potential_fall_frames > 0:
            st.write(f"However, {potential_fall_frames} frames showed potential fall-like behavior.")
        if standing_frames == frame_count:
            st.write("The video mostly shows normal standing/walking behavior.")
        else:
            st.write("The video may contain some unstable movements, but no clear falls.")

    if transition_count > 0:
        st.write(f"On average, a state change (fall or recovery) occurred every {frame_count/transition_count:.2f} frames.")

def process_bag_file(bag_file_path, rf_model, dl_model, net):
    st.write(f"Processing bag file: {bag_file_path}")

    pipeline = rs.pipeline()
    config = rs.config()

    rs.config.enable_device_from_file(config, bag_file_path)

    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

    pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    frame_count = 0
    fall_frames = 0
    standing_frames = 0
    transition_count = 0
    previous_state = "Standing"
    fall_detected = False
    fall_start_frame = None
    potential_fall_frames = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            frame_count += 1
            features = extract_features(color_image)

            y_pred_rf = rf_model.predict(features)[0]
            y_pred_dl = (dl_model.predict(features) > 0.5).astype("int32")[0][0]

            fall_likelihood = (y_pred_rf + y_pred_dl) / 2

            height_width_ratio, _, vertical_position, _, _, _, velocity, acceleration = features[0]

            if height_width_ratio < 1.2 or vertical_position > 0.7 or (abs(velocity) > 50 and abs(acceleration) > 100):
                potential_fall_frames += 1
            else:
                potential_fall_frames = max(0, potential_fall_frames - 1)

            if potential_fall_frames > 5 or fall_likelihood > 0.5:
                current_state = "Fallen"
                fall_frames += 1
                if previous_state != "Fallen":
                    fall_detected = True
                    transition_count += 1
                    fall_start_frame = frame_count
            else:
                current_state = "Standing"
                standing_frames += 1

            if fall_detected:
                info_placeholder.error(f"FALL DETECTED at frame {fall_start_frame}!")
                fall_detected = False

            info_placeholder.write(f"Current State: {current_state}")
            info_placeholder.write(f"Fall Likelihood: {fall_likelihood:.2f}")
            info_placeholder.write(f"Potential Fall Frames: {potential_fall_frames}")
            info_placeholder.write(f"Features: {features[0]}")

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))
            frame_placeholder.image(images, caption='Color and Depth Frames', channels="RGB")

            previous_state = current_state
            time.sleep(0.033)  # Adjust this value to change the playback speed

    except RuntimeError:
        st.write("Reached the end of the bag file.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        pipeline.stop()

    # Provide summary
    st.write("Bag File Processing Complete. Summary:")
    st.write(f"Total frames processed: {frame_count}")
    st.write(f"Frames classified as standing: {standing_frames} ({standing_frames/frame_count*100:.2f}%)")
    st.write(f"Frames classified as fallen: {fall_frames} ({fall_frames/frame_count*100:.2f}%)")
    st.write(f"Number of transitions detected: {transition_count}")

    if fall_frames > 0:
        st.write("The bag file contains fall events.")
        if transition_count > 1:
            st.write("Multiple fall/recovery events were detected.")
        elif fall_frames == frame_count:
            st.write("The entire recording shows a fallen state.")
        else:
            st.write(f"A fall event was detected starting at frame {fall_start_frame}.")
    else:
        st.write("No definitive falls were detected in this recording.")
        if potential_fall_frames > 0:
            st.write(f"However, {potential_fall_frames} frames showed potential fall-like behavior.")
        if standing_frames == frame_count:
            st.write("The recording mostly shows normal standing/walking behavior.")
        else:
            st.write("The recording may contain some unstable movements, but no clear falls.")

    if transition_count > 0:
        st.write(f"On average, a state change (fall or recovery) occurred every {frame_count/transition_count:.2f} frames.")